import ast
import csv
import json
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable, List
from .schema import CanonicalSample, Question
LOG = logging.getLogger('evaluate_all')
OVO_TASK_GROUPS = {'backward': ['EPM', 'ASI', 'HLD'], 'realtime': ['STU', 'OJR', 'ATR', 'ACR', 'OCR', 'FPD'], 'forward': ['REC', 'SSR', 'CRR']}
STREAMING_TASK_CODES = {'Object Perception': 'OP', 'Action Perception': 'AP', 'Text-Rich Understanding': 'TRU', 'Clips Summarize': 'CS', 'Attribute Perception': 'ATP', 'Spatial Understanding': 'SPU', 'Counting': 'CNT', 'Event Understanding': 'EU', 'Causal Reasoning': 'CR', 'Prospective Reasoning': 'PR', 'Emotion Recognition': 'ER', 'Multimodal Alignment': 'MA', 'Scene Understanding': 'SCU', 'Source Discrimination': 'SD', 'Misleading Context Recognition': 'MCR', 'Anomaly Context Understanding': 'ACU', 'Sequential Question Answering': 'SQA', 'Proactive Output': 'PO', 'Active Output': 'PO'}

def parse_options(value) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except (SyntaxError, ValueError):
        pass
    return [item.strip() for item in str(value).split('|') if item.strip()]

def parse_timestamp(value) -> float:
    if value is None or value == '':
        return 0.0
    parts = str(value).split(':')
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(value)

def option_letter(value: str) -> str:
    match = re.search('\\b([A-Z])\\b', str(value).upper())
    return match.group(1) if match else str(value).strip().upper()

class BenchmarkAdapter:
    name = ''
    mode = 'offline'

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._selected_question_ids = set()
        self._last_samples = []

    def load_samples(self) -> Iterable[CanonicalSample]:
        raise NotImplementedError

class OVOBenchAdapter(BenchmarkAdapter):
    name = 'ovobench'
    mode = 'streaming'

    def load_samples(self):
        path = Path(self.config['annotation'])
        data = json.loads(path.read_text(encoding='utf-8'))
        root = Path(self.config.get('chunked_dir', 'OVO-Bench/data/chunked_videos'))
        allowed_video_ids = self.config.get('_video_ids')
        if allowed_video_ids is not None:
            allowed_video_ids = {str(v) for v in allowed_video_ids}
        for item in data:
            if allowed_video_ids is not None and str(item['id']) not in allowed_video_ids:
                continue
            task = item['task']
            questions = []
            if task in {'REC', 'SSR', 'CRR'}:
                for index, test in enumerate(item.get('test_info', [])):
                    questions.append(Question(question_id=f"{item['id']}:{index}", text='', timestamp=float(test['realtime']), task_type=task, answer=test.get('count', test.get('type')), response_policy='may_defer', metadata={'test_index': index, 'test_info': test}))
            else:
                questions.append(Question(question_id=str(item['id']), text=item['question'], options=item.get('options', []), timestamp=float(item.get('realtime', 0.0)), task_type=task, answer=chr(65 + int(item['gt'])), response_policy='reply_required'))
            first_video = f"{item['id']}_{questions[0].metadata['test_index']}.mp4" if task in {'REC', 'SSR', 'CRR'} else f"{item['id']}.mp4"
            yield CanonicalSample(benchmark=self.name, sample_id=str(item['id']), video_path=str(root / first_video), questions=questions, mode='streaming', metadata=item)

class StreamingBenchAdapter(BenchmarkAdapter):
    name = 'streamingbench'
    mode = 'streaming'

    @staticmethod
    def _category_name(category):
        name = category.lower()
        if name.startswith('real_time'):
            return 'real_time_visual'
        if name.startswith('omni_source'):
            return 'omni_source'
        if name.startswith('contextual') or name.startswith('sequential'):
            return 'contextual'
        if name.startswith('proactive'):
            return 'proactive'
        return name

    @staticmethod
    def _video_id(qid):
        parts = qid.rsplit('_', 2)
        if len(parts) >= 3 and parts[-2].isdigit():
            return parts[-2]
        return qid

    @staticmethod
    def _video_subdir(category, task_type):
        category_name = str(category).lower()
        task_name = str(task_type).lower()
        if category_name.startswith('real_time'):
            return Path('raw/real')
        if category_name.startswith('sequential'):
            return Path('raw/sqa')
        if category_name.startswith('proactive'):
            return Path('raw/proactive')
        omni_dirs = {'anomaly context understanding': 'anomaly', 'misleading context recognition': 'misleading', 'misleading context understanding': 'misleading', 'emotion recognition': 'emotion', 'multimodal alignment': 'multimodal', 'scene understanding': 'scene', 'source discrimination': 'source'}
        for task_name_key, subdir in omni_dirs.items():
            if task_name == task_name_key:
                return Path('raw/omni') / subdir
        raise ValueError(f'Cannot map StreamingBench video directory: category={category!r}, task_type={task_type!r}')

    def load_samples(self):
        root = Path(self.config.get('video_root', 'datasets/StreamingBench'))
        annotations = self.config.get('annotations', {})
        if self.config.get('input_modality', 'visual_only') == 'visual_only':
            LOG.warning('StreamingBench Omni-Source tasks ER/SCU/SD/MA are evaluated with visual-only input; audio is not provided.')
        for category, annotation in annotations.items():
            path = Path(annotation)
            is_sqa = category.lower().startswith('sequential')
            allowed_video_ids = self.config.get('_video_ids')
            if allowed_video_ids is not None:
                allowed_video_ids = {str(v) for v in allowed_video_ids}
            grouped = {}
            order = []
            with path.open('r', encoding='utf-8', newline='') as handle:
                for row in csv.DictReader(handle):
                    qid = row['question_id']
                    video_id = self._video_id(qid)
                    if allowed_video_ids is not None and str(video_id) not in allowed_video_ids:
                        continue
                    task_type = row.get('task_type', category)
                    video_path = root / self._video_subdir(category, task_type) / f'sample_{video_id}' / 'video.mp4'
                    is_proactive = category.lower().startswith('proactive')
                    timestamp_key = 'ground_truth_time_stamp' if is_proactive else 'time_stamp'
                    policy = 'standing_trigger' if is_proactive else 'reply_required'
                    question = Question(question_id=qid, text=row['question'], options=parse_options(row.get('options', [])), timestamp=parse_timestamp(row.get('time_stamp')), task_type=task_type, answer=row.get('answer', row.get('ground_truth_output')), response_policy=policy, metadata={**row, 'category': category, 'video_id': video_id, 'category_name': self._category_name(category), 'task_code': STREAMING_TASK_CODES.get(row.get('task_type', category), row.get('task_type', category)), 'ground_truth_timestamp': parse_timestamp(row.get(timestamp_key)) if row.get(timestamp_key) else None})
                    if is_sqa:
                        if video_id not in grouped:
                            grouped[video_id] = []
                            order.append(video_id)
                        grouped[video_id].append((video_path, question))
                    else:
                        yield CanonicalSample(benchmark=self.name, sample_id=qid, video_path=str(video_path), questions=[question], mode='streaming', metadata={'category': category, 'category_name': self._category_name(category), 'context_mode': self.config.get('context_mode', 'main_60s'), 'proactive_protocol': self.config.get('proactive_protocol', 'official_polling'), 'input_modality': self.config.get('input_modality', 'visual_only')})
            if is_sqa:
                for video_id in order:
                    values = sorted(grouped[video_id], key=lambda item: item[1].timestamp or 0.0)
                    yield CanonicalSample(benchmark=self.name, sample_id=f'sqa:{video_id}', video_path=str(values[0][0]), questions=[item[1] for item in values], mode='streaming', metadata={'category': category, 'category_name': self._category_name(category), 'context_mode': self.config.get('context_mode', 'main_60s'), 'proactive_protocol': self.config.get('proactive_protocol', 'official_polling'), 'input_modality': self.config.get('input_modality', 'visual_only')})
