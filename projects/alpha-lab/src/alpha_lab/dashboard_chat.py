"""Explicitly configured public-question chat; never reads an experiment workspace."""
from __future__ import annotations

from dataclasses import asdict,fields
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import threading
from urllib.parse import urlsplit

from alpha_lab.client import get_provider
from alpha_lab.llm_runtime import LLMRunError,LLMRuntime,LLMSettings
from alpha_lab.market_data.sharadar_store import atomic_json

PUBLIC_SYSTEM = (
    "Explain general US equity research methods and public factor concepts. "
    "Only the submitted question is available. You have no files, tools, datasets, "
    "experiment results or live workspace status. Direct local-status questions "
    "to the dashboard's status and experiment panels. Distinguish economic "
    "hypotheses from validated investment results. Keep the explanation concise."
)


class ChatError(RuntimeError):
    def __init__(self,message,status_code):
        super().__init__(message)
        self.status_code=status_code


class PublicChatSession:
    """One immutable intent and shared budget for the viewer's configured session."""
    def __init__(self,config_path:Path):
        config_path=Path(config_path)
        if config_path.stat().st_size>65_536:
            raise ValueError("Chat config exceeds its size limit")
        config=json.loads(config_path.read_text(encoding='utf-8'))
        if not isinstance(config,dict) or set(config)!={'data_scope','base_url','run_directory','max_question_bytes','llm'}:
            raise ValueError("Chat config requires exactly the documented fields")
        if config['data_scope']!='public_only':
            raise ValueError("Automatic workspace context is not supported by public chat")
        selected=config['llm']
        if not isinstance(selected,dict) or set(selected)!={field.name for field in fields(LLMSettings)}:
            raise ValueError("Intentionally set every LLM control in chat config")
        self.settings=LLMSettings(**selected)
        if self.settings.deadline_utc is None:
            raise ValueError("Chat requires an explicit UTC authorization cutoff")
        if datetime.fromisoformat(self.settings.deadline_utc)<=datetime.now(timezone.utc):
            raise ValueError("Chat authorization cutoff already passed")
        self.base_url=config['base_url']
        endpoint=urlsplit(self.base_url) if isinstance(self.base_url,str) else None
        if (endpoint is None or endpoint.scheme!='https' or not endpoint.netloc
                or endpoint.username is not None or endpoint.password is not None
                or endpoint.query or endpoint.fragment):
            raise ValueError("Chat base URL must be an explicit HTTPS endpoint without credentials")
        self.max_question_bytes=config['max_question_bytes']
        if (isinstance(self.max_question_bytes,bool) or not isinstance(self.max_question_bytes,int)
                or not 0<self.max_question_bytes<=self.settings.max_context_bytes):
            raise ValueError("Set a positive question byte limit within the context cap")
        minimum_context=len(json.dumps([PUBLIC_SYSTEM,[{'role':'user','content':'?'}],[]],
                                       ensure_ascii=False).encode('utf-8'))
        self._minimum_reservation=minimum_context+8192+self.settings.max_output_tokens
        if self.settings.max_context_bytes<minimum_context or self.settings.max_reserved_tokens<self._minimum_reservation:
            raise ValueError("Context or reserved-token budget cannot fund one question")
        directory=config['run_directory']
        if not isinstance(directory,str) or not directory.strip():
            raise ValueError("Select a new immutable chat run directory")
        directory=Path(directory)
        self.directory=(directory if directory.is_absolute() else config_path.parent/directory).resolve()
        self.directory.mkdir(parents=True,exist_ok=False)
        self.runtime=LLMRuntime(self.settings,self.directory/'usage.json')
        self.runtime.save()
        self._provider=None
        self._lock=threading.Lock()
        self._state_lock=threading.Lock()
        self._closed=False
        self._intent={
            'created_utc':datetime.now(timezone.utc).isoformat(),'data_scope':'public_only',
            'base_url':self.base_url,'llm':asdict(self.settings),'max_question_bytes':self.max_question_bytes,
            'workspace_context':False,'tools':[],'history':'Each question independent; shared session budget',
            'sdk_retries':0,'automatic_retries':0,'concurrency':1,'store':False,
            'retention':'Gateway pricing and retention unknown; store=false is not a retention guarantee',
            'temperature':'intentionally omitted','top_p':'intentionally omitted',
            'logging':'Controls, status and token usage only; questions and answers are not saved',
            'context':'Explicit byte cap, independent questions, no automatic truncation or summaries',
            'structured_output':'Plain text answer; no tool execution',
            'seeds':'Not applicable: explanatory questions, no experiments; generation is not deterministic',
            'cost_budget':'Shared call/token/time caps; gateway pricing unknown, no dollar estimate',
            'run_clock':'Begins at session configuration; mandatory absolute UTC cutoff',
            'status':'configured','calls':0,
        }
        atomic_json(self.directory/'intent.json',self._intent)
        atomic_json(self.directory/'run.json',self._intent)

    def _record_status(self,status):
        with self._state_lock:
            self._intent.update(status='closed' if self._closed else status,calls=self.runtime.calls)
            atomic_json(self.directory/'run.json',self._intent)

    def describe(self):
        used=max(self.runtime.reserved_tokens,self.runtime.input_tokens+self.runtime.output_tokens)
        available=(not self._closed and self.runtime.remaining_seconds()>0 and self.runtime.calls<self.settings.max_calls
                   and used+self._minimum_reservation<=self.settings.max_reserved_tokens
                   and os.environ.get('OPENAI_BASE_URL','').rstrip('/')==self.base_url.rstrip('/')
                   and bool(os.environ.get('OPENAI_API_KEY','').strip()))
        return {'enabled':available,'data_scope':'public_only','workspace_context':False,
                'base_url':self.base_url,'model':self.settings.model,'reasoning_effort':self.settings.reasoning_effort,
                'max_question_bytes':self.max_question_bytes,'controls':asdict(self.settings),
                'calls':self.runtime.calls,'remaining_calls':max(0,self.settings.max_calls-self.runtime.calls),
                'reserved_tokens':self.runtime.reserved_tokens,
                'reason':None if available else 'Chat unavailable: session limits, local credentials or endpoint configuration need review.'}

    def answer(self,question:str):
        if not isinstance(question,str) or not question.strip():
            raise ChatError("Enter a public research question.",400)
        question=question.strip()
        if len(question.encode('utf-8'))>self.max_question_bytes:
            raise ChatError("Question exceeds the configured byte limit.",413)
        known_keys=[value for name,value in os.environ.items() if name.endswith('API_KEY') and len(value)>=8]
        if any(value in question for value in known_keys):
            raise ChatError("API credentials are not valid research questions.",400)
        if not self._lock.acquire(blocking=False):
            raise ChatError("A research question is already running. Wait for its result.",429)
        try:
            try:
                with self._state_lock:
                    if self._closed or self.runtime.remaining_seconds()<=0:
                        raise ChatError("Chat session closed or authorization cutoff reached.",410)
                    if self.runtime.calls>=self.settings.max_calls:
                        raise ChatError("Shared chat call budget reached.",410)
                    used=max(self.runtime.reserved_tokens,self.runtime.input_tokens+self.runtime.output_tokens)
                    if used+self._minimum_reservation>self.settings.max_reserved_tokens:
                        raise ChatError("Shared chat token budget reached.",410)
                    if os.environ.get('OPENAI_BASE_URL','').rstrip('/')!=self.base_url.rstrip('/'):
                        raise ChatError("Configured endpoint differs from OPENAI_BASE_URL; update the local configuration.",503)
                    if not os.environ.get('OPENAI_API_KEY','').strip():
                        raise ChatError("Configure OPENAI_API_KEY locally before sending a question.",503)
                    if self._provider is None:
                        self._provider=get_provider('openai',runtime=self.runtime)
                answer=self._provider.complete(model=self.settings.model,system=PUBLIC_SYSTEM,
                                               messages=[{'role':'user','content':question}],
                                               max_tokens=self.settings.max_output_tokens)
                if not isinstance(answer,str) or not answer.strip():
                    raise LLMRunError("Model returned no answer")
                if any(value in answer for value in known_keys):
                    raise LLMRunError("Answer contains an API credential")
            except ChatError:
                raise
            except Exception:
                self._record_status('request_failed')
                raise ChatError("Research request failed or a configured limit was reached. No automatic retry.",502) from None
            self._record_status('request_completed')
            if self._closed:
                raise ChatError("Chat session closed while the question was running.",410)
            return answer
        finally:
            self._lock.release()

    def close(self):
        with self._state_lock:
            self._closed=True
            provider=self._provider
        try:
            if provider is not None:
                provider.close()
        finally:
            self._record_status('closed')
