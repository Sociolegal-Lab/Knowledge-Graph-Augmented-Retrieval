
OPENAI_API_KEY = ""
import json
from openai import OpenAI
client = OpenAI(
    api_key=OPENAI_API_KEY
)

upload_file = "lbox_open_casename\\lbox.jsonl"
description = "trachi"

batch_input_file = client.files.create(
  file=open(upload_file, "rb"),
  purpose="batch"
)

batch_input_file_id = batch_input_file.id

print("Batch input file created with id:", batch_input_file_id)

batch_response = client.batches.create(
    input_file_id=batch_input_file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
      "description": description
    }
)
