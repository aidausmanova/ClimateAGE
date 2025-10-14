import os
import numpy
import numpy as np
import torch
import torch.cuda
from typing import Tuple, List
import transformers

from vllm import LLM, SamplingParams


from ..prompts.prompt_template_manager import *
from ..utils.basic_utils import *
from ..utils.consts import *
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

from transformers import PreTrainedTokenizer

def lm_instruction(instruction):
    return "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"


# class VLLMLlama:
#     def __init__(self, model_name, cache_dir=None, cache_filename=None, max_model_len=4096, **kwargs):
#         # model_name = 'meta-llama/Llama-3.3-70B-Instruct'
        
#         pipeline_parallel_size = 1
#         tensor_parallel_size = torch.cuda.device_count()
        
#         # os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
#         self.model_name = model_name
#         self.client = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size, pipeline_parallel_size=pipeline_parallel_size,
#                           seed=0, dtype='auto', max_seq_len_to_capture=max_model_len, enable_prefix_caching=True,
#                           enforce_eager=False, gpu_memory_utilization=0.8,
#                           max_model_len=max_model_len, quantization=None, load_format='auto', trust_remote_code=True)
        
#         self.tokenizer = self.client.get_tokenizer()
    
#     def infer(self, messages: List[TextChatMessage], max_tokens=2048):
#         logger.info(f"Calling VLLM offline, # of messages {len(messages)}")
#         messages_list = [messages]
#         prompt_ids = convert_text_chat_messages_to_input_ids(messages_list, self.tokenizer)

#         vllm_output = self.client.generate(prompt_token_ids=prompt_ids,  sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0))
#         response = vllm_output[0].outputs[0].text
#         prompt_tokens = len(vllm_output[0].prompt_token_ids)
#         completion_tokens = len(vllm_output[0].outputs[0].token_ids )
#         metadata = {
#             "prompt_tokens": prompt_tokens,
#             "completion_tokens": completion_tokens
#         }
#         return response, metadata

#     def batch_infer(self, messages_list: List[List[TextChatMessage]], max_tokens=2048, json_template=None):
#         if len(messages_list) > 1:
#             logger.info(f"Calling VLLM offline, # of messages {len(messages_list)}")

#         guided = None
#         if json_template is not None:
#             from vllm.model_executor.guided_decoding.guided_fields import GuidedDecodingRequest
#             guided = GuidedDecodingRequest(guided_json=PROMPT_JSON_TEMPLATE[json_template])

#         all_prompt_ids = [convert_text_chat_messages_to_input_ids(messages, self.tokenizer) for messages in messages_list]
#         vllm_output = self.client.generate(prompt_token_ids=all_prompt_ids,
#                                            sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0),
#                                            guided_options_request=guided)

#         all_responses = [completion.outputs[0].text for completion in vllm_output]
#         all_prompt_tokens = [len(completion.prompt_token_ids) for completion in vllm_output]
#         all_completion_tokens = [len(completion.outputs[0].token_ids) for completion in vllm_output]

#         metadata = {
#             "prompt_tokens": sum(all_prompt_tokens),
#             "completion_tokens": sum(all_completion_tokens),
#             "num_request": len(messages_list)
#         }
#         return all_responses, metadata
    

class InfoExtractor:
    def __init__(
        self,
        exp="base",
        engine="Llama-3.3-70B-Instruct",
        n_shot=10,
        use_vllm=True,
    ):
        if engine == "NA":
            return
        if exp == "0_shot":
            self.PROMPT_TEMPLATE = PROMPT_TEMPLATE_ZERO_SHOT
        elif exp == "no_rag":
            self.PROMPT_TEMPLATE = PROMPT_TEMPLATE_NO_RAG
        elif exp == "no_relation":
            self.PROMPT_TEMPLATE = PROMPT_TEMPLATE_NO_RELATION
        else:
            print(f"Using base prompt template for exp = {exp}")
            self.PROMPT_TEMPLATE = PROMPT_TEMPLATE
        if "_shot" in exp:
            n_shot = int(exp.split("_")[0])

        self.model = None
        self.use_vllm = use_vllm

        # Load LLM model
        model_id = "meta-llama/" + engine
        print(f"Loading model from {model_id}")

        if self.use_vllm:
            from vllm import LLM, SamplingParams

            # https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct/blob/main/generation_config.json
            self.sampling_params = SamplingParams(
                temperature=0.6,
                top_p=0.9,
                stop=[DELIMITERS["completion_delimiter"]],
                max_tokens=2048,
            )
            self.model = LLM(
                model=model_id,
                task="generate",
                tensor_parallel_size=1,
                gpu_memory_utilization=0.8,
                max_model_len=20000, # 4096,  # 25390,
                enable_prefix_caching=True,
            )
        else:
            # --------- Huggingface --------
            self.model = transformers.pipeline(
                "text-generation",
                model=model_id,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
            )
            self.processor = [
                self.model.tokenizer.eos_token_id,
                self.model.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ]
            self.model.tokenizer.pad_token_id = self.model.tokenizer.eos_token_id

        # load n-shot examples
        examples = load_json_file(PATH["LLM"]["examples"])
        self.formatted_examples = ""
        print("# of examples: ", len(examples))
        for i, example in enumerate(examples[:n_shot]):
            if exp == "no_rag":
                example = re.sub(
                    r"\nPotential Entities:.*?\n##",
                    "\n##",
                    example,
                    flags=re.DOTALL,
                )
            if exp == "no_relation":
                example = re.sub(
                    r'##\n\("relationship"<\|>.*?\)\n', "", example, flags=re.DOTALL
                )
            self.formatted_examples += f"\nExample {i+1}:\n{example}"

    def parse_response(self, response, with_description=True):
        out = {"entities": [], "relationships": []}
        # trim the response to start from the first entity
        start_index = response.find('("')
        if start_index != 0:
            response = response[start_index:]

        # trim the response to end with <|COMPLETE|>
        # if not self.use_vllm:
        response = response.split(DELIMITERS["completion_delimiter"])[0]

        # split response into records
        response = response.split(DELIMITERS["record_delimiter"])
        response = [
            r.lstrip("\n").rstrip("\n").lstrip("(").rstrip(")") for r in response
        ]

        # split the response into items
        pattern = r"<\s*\|\s*>"
        response = [re.split(pattern, r) for r in response]
        for r in response:
            if "entity" in r[0]:
                if with_description:
                    if len(r) == 4:
                        out["entities"].append(
                            {"name": r[1], "label": r[2], "description": r[3]}
                        )
                elif len(r) == 3:
                    out["entities"].append({"name": r[1], "label": r[2]})

            elif "relationship" in r[0]:
                if len(r) == 4:
                    out["relationships"].append(
                        {"source": r[1], "target": r[2], "relation": r[3]}
                    )
        return out

    def run(
        self,
        text,
        retrieved_nodes,
    ):

        potential_entities = list(retrieved_nodes.keys())
        potential_entities = ", ".join(potential_entities)

        prompt = self.PROMPT_TEMPLATE.format(
            **DELIMITERS,
            formatted_examples=self.formatted_examples,
            input_text=text.replace("{", "").replace("}", ""),
            potential_entities=potential_entities,
        ).format(**DELIMITERS)

        conversation = [{"role": "user", "content": prompt}]
        if self.model:
            if self.use_vllm:
                output = self.model.generate(
                    prompt, sampling_params=self.sampling_params
                )
                pred_content = output[0].outputs[0].text
            else:
                outputs = self.model(
                    prompt,
                    max_new_tokens=4000,
                    pad_token_id=self.model.tokenizer.eos_token_id,
                    # eos_token_id=terminators,
                    do_sample=True,
                    return_full_text=False,
                    temperature=0.6,
                    batch_size=8,
                    # top_p=0.9,
                )
                # pred_content = outputs[0]["generated_text"][-1]["content"]
                pred_content = outputs[0]["generated_text"]

        conversation.append({"role": "assistant", "content": pred_content})
        response = self.parse_response(pred_content)
        return response, conversation
