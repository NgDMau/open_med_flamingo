import json, requests

url = "http://0.0.0.0:1234/clever_flamingo"

instruction = "Can you give some details in the picture? Please be correct, don't say something you are not sure"
# instruction = "What is capital of France?"
# instruction = "\n\n+ Reasoning:\n- Step 1: Recognize the disease area\n- Answer: ```json\n[{\"bbox_2d\": [37, 242, 122, 405], \"label\": \"disease area\"}, {\"bbox_2d\": [113, 244, 199, 400], \"label\": \"disease area\"}]\n```\n- Step 2: How would you specify the visible traits of this lesion?\n- Answer: Substantial sulcal widening of posterior cingulate and parieto-occipital sulci, substantial gyral atrophy.\n- Step 3: What is the stage of the lesion?\n- Answer: Koedam = 2\n\n+ Final Answer:"
instruction = """What's the clinical severity of this disease state?"""
content_lst = {
    # remenber to add '<image>' to your instruction to indecate the location of image(s)
    "prompt": f"{instruction}\n",
    "imgpaths": ["/app/baseline_models/sample_data/llama_mri_cot/images/test/238.jpg"],
    "args": {
        "max_new_token": 512,
        "num_beams": 1,
        "temperature": 1.0,
        "top_k": 20,
        "top_p": 1,
        "do_sample": True,
        "length_penalty": 1.0,
        "no_repeat_ngram_size": 3,
    },
}

d = {"content_lst": content_lst, "typ": "None"}
d = json.dumps(d).encode("utf8")
r = requests.post(url, data=d)
js = json.loads(r.text)

print(js["result"]["response"])
