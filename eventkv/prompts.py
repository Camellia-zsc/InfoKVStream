BACKWARD_TASKS = ['EPM', 'ASI', 'HLD']
REAL_TIME_TASKS = ['OCR', 'ACR', 'ATR', 'STU', 'FPD', 'OJR']
FORWARD_TASKS = ['REC', 'SSR', 'CRR']
BR_PROMPT_TEMPLATE = '\nQuestion: {}\nOptions:\n{}\n\nRespond only with the letter corresponding to your chosen option (e.g., A, B, C). \nDo not include any additional text or explanation in your response.\n'
REC_PROMPT_TEMPLATE = "\nYou're watching a video in which people may perform a certain type of action repetively. \nThe person performing this kind of action are referred to as 'they' in the following statement.\nYou're task is to count how many times have different people in the video perform this kind of action in total.\nOne complete motion counts as one. \nNow, answer the following question: {}\nProvide your answer as a single number (e.g., 0, 1, 2, 3…) indicating the total count.\nDo not include any additional text or explanation in your response.\n"
SSR_PROMPT_TEMPLATE = "\nYou're watching a tutorial video which contain a sequential of steps. \nThe following is one step from the whole procedures: \n{}\nYour task is to determine if the man or woman in the video is currently performing this step.\nAnswer only with “Yes” or “No”.\nDo not include any additional text or explanation in your response.\n"
CRR_PROMPT_TEMPLATE = "\nYou're responsible of answering questions based on the video content. \nThe following question are relevant to the latest frames, i.e. the end of the video.\n{}\nDecide whether existing visual content, especially latest frames, i.e. frames that near the end of the video, provide enough information for answering the question.\nAnswer only with “Yes” or “No”.\nDo not include any additional text or explanation in your response.\n"

def build_prompt(task, question, options, _anno_, index):
    if task in ['EPM', 'ASI', 'HLD', 'STU', 'OJR', 'ATR', 'ACR', 'OCR', 'FPD']:
        formatted_options = '; '.join((f'{chr(65 + i)}. {option}' for i, option in enumerate(options))) + ';'
        prompt = BR_PROMPT_TEMPLATE.format(question, formatted_options)
    elif task == 'REC':
        activity = _anno_['activity']
        question = 'How many times did they ' + activity + '?'
        prompt = REC_PROMPT_TEMPLATE.format(question)
    elif task == 'SSR':
        step = _anno_['test_info'][index]['step']
        prompt = SSR_PROMPT_TEMPLATE.format(step)
    elif task == 'CRR':
        question = _anno_['question']
        prompt = CRR_PROMPT_TEMPLATE.format(question)
    else:
        raise ValueError(f'Unsupported task: {task}')
    return prompt
