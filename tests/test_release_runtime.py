"""Standalone real tiny Qwen integration and benchmark protocol checks."""
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
import yaml
from PIL import Image
from eventkv.model import EventKVModel
from eventkv.qwen_model import EventKVQwen
from eventkv.signals import incoming_attention_scores
from eventkv.evaluate import jobs_for, correct, metrics
ROOT=Path(__file__).resolve().parents[1]

class Tokenizer:
    eos_token_id=31
    pad_token_id=0
    def __call__(self,text,**kwargs):
        ids=[1+ord(c)%29 for c in str(text)] or [1]
        return SimpleNamespace(input_ids=torch.tensor([ids]) if kwargs.get('return_tensors')=='pt' else ids)
    def decode(self,ids,**kwargs):return 'A' if 7 in [int(i) for i in ids] else ''

class Head(torch.nn.Module):
    def forward(self,hidden):
        logits=hidden.new_zeros((*hidden.shape[:-1],32));logits[...,7]=1
        return logits

def tiny_model():
    from transformers import Qwen2VLImageProcessor
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
    cfg=Qwen2_5_VLConfig(text_config=dict(hidden_size=32,intermediate_size=64,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,vocab_size=32,
        rope_scaling={'rope_type':'default','mrope_section':[1,1,2]},layer_types=['full_attention']*2),
        vision_config=dict(depth=2,hidden_size=16,intermediate_size=32,num_heads=4,patch_size=2,
            spatial_merge_size=2,temporal_patch_size=2,window_size=8,out_hidden_size=32,fullatt_block_indexes=[0,1]))
    cfg._attn_implementation='sdpa'
    torch.manual_seed(731)
    qa=EventKVQwen(cfg).eval()
    class Processor:
        tokenizer=Tokenizer()
        def __init__(self):
            self.images=Qwen2VLImageProcessor(min_pixels=16,max_pixels=256,patch_size=2,temporal_patch_size=2,merge_size=2)
        def __call__(self,*,images,text,return_tensors):return self.images(images=images,return_tensors=return_tensors)
    qa.processor=Processor();qa.lm_head=Head()
    qa.num_layers=2;qa.init_prompt_ids=torch.tensor([[1,2,3]])
    qa._position_ids_cache=[None]*2;qa._layer_position_ids={}
    qa.conv_history=[];qa.kv_cache=None
    return qa

@pytest.mark.parametrize('version',['initial_improvement','latest'])
def test_standalone_real_visual_to_answer_and_second_video(version):
    cfg=yaml.safe_load((ROOT/f'configs/{version}.yaml').read_text())
    cfg['eventkv'].update(recent_budget=16,recent_fallback_budget=16,event_budget=12,
        kv_budget=4,text_budget=100,event_text_max_tokens=8,summary_max_new_tokens=2,summary_cost_threshold=0)
    qa=tiny_model();model=EventKVModel(cfg,qa_model=qa,processor=qa.processor)
    try:
        for repeat in range(2):
            model.reset()
            for i in range(12):
                model.stream_video_chunk([Image.new('RGB',(16,16),(255*(i%2),0,0))]*2,seconds=4.)
            assert model.ask_question('Which event happened?',max_new_tokens=2)=='A'
            assert max(k.shape[-2] for k,v in qa.kv_cache)<=7
            model.snapshot()
            assert model.last_stats['kv_budget']==4
            assert model.last_stats['summary_generation_calls']>0
            assert max(model.last_stats['persistent_visual_kv_tokens_by_layer'])<=4
            assert len(model.controller.memory.short)==0
            assert all(not k.requires_grad for k,v in qa.kv_cache)
    finally:model.close()

@pytest.mark.parametrize('causal',[False,True])
def test_grouped_signal_matches_explicit_repeated_keys(causal):
    torch.manual_seed(31)
    q=torch.randn(2,4,5,8);k=torch.randn(2,2,9,8);qp=torch.tensor([0,2,4,6,8]);kp=torch.arange(9)
    scores,zero=incoming_attention_scores(q,k,qp,kp,causal=causal,query_block_size=2)
    logits=q@k.repeat_interleave(2,dim=1).transpose(-1,-2)/(8**.5)
    if causal:logits.masked_fill_(kp[None,:]>qp[:,None],-torch.inf)
    torch.testing.assert_close(scores,logits.softmax(-1).mean((0,1,2)))
    assert zero==0

def test_sqa_ground_truth_history(tmp_path):
    p=tmp_path/'sqa.csv'
    p.write_text('question_id,question,options,time_stamp,answer\nsqa_1_1,First?,"[\'A. x\', \'B. y\']",00:01,A\nsqa_1_2,Second?,"[\'A. x\', \'B. y\']",00:02,B\n')
    jobs=list(jobs_for('streamingbench',dict(video_root=str(tmp_path),annotations={'sequential':str(p)},context_mode='full_prefix')))
    assert len(jobs)==2
    assert 'Question: First?; Answer: A;' in jobs[1]['prompt']
    assert jobs[0]['start_time']==0
    assert jobs[1]['prompt'].endswith('Please select the correct answer.')

def test_official_ovo_scoring_and_incomplete_coverage():
    base={'benchmark':'ovobench','task_type':'SSR','prediction':'No','ground_truth':0}
    assert correct(base)
    assert not correct({**base,'ground_truth':1})
    assert correct({**base,'task_type':'REC','prediction':'12 times','ground_truth':12})
    out=metrics([base],2)
    assert out['complete'] is False and out['ovo_overall_macro'] is None
