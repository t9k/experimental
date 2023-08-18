import os

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import asyncio


async def homepage(request):
    payload = await request.body()
    string = payload.decode("utf-8")
    response_q = asyncio.Queue()
    await request.app.model_queue.put((string, response_q))
    output = await response_q.get()
    return JSONResponse(output)


async def server_loop(q):
    path = os.getenv('MODEL_PATH')
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path)
    pipe = pipeline(task='text-generation',
                    model=model,
                    tokenizer=tokenizer,
                    device=0)
    while True:
        (string, response_q) = await q.get()
        out = pipe(string)
        await response_q.put(out)


app = Starlette(routes=[
    Route("/", homepage, methods=["POST"]),
], )


@app.on_event("startup")
async def startup_event():
    q = asyncio.Queue()
    app.model_queue = q
    asyncio.create_task(server_loop(q))
