# Project 26: Tool Use and Function Calling

## Hook

Your **Project 21: Fine-Tuning and Instruction Tuning** model can write a poem about the weather. It cannot tell you the actual weather. Ask it for today's temperature in Toronto and it will cheerfully invent a number that sounds plausible, because inventing tokens is the only thing it knows how to do. A language model produces text. The weather lives behind an HTTP request. There is no part of the model that can reach out and call a Python function, wait for a result, and read the response. How does a thing that only emits tokens get to invoke a function and use what comes back?

## The Concept

Imagine a chess player who is not allowed to touch the pieces. The player sits on one side of a screen and calls out moves to a referee. "Knight to f3." The referee picks up the piece, moves it, then shows the player the new board. The player thinks again, calls out the next move, and the referee executes. The player is the planner. The referee is the executor. Neither one does the other's job. The interesting trick is the protocol: what the player is allowed to say, and what the referee is allowed to do in response.

Tool use is that protocol applied to a language model. The model is the player. A small runtime sitting around the model is the referee. The model is trained to emit a specific format when it wants something done, a JSON blob naming a function and its arguments. The runtime watches the model's output, recognizes the format, parses the JSON, calls the function the model named, captures the result, and feeds the result back to the model as a new message. The model then resumes generating.

That is the whole design. A language model that can use tools is a language model with two new behaviors layered on top: it has learned to emit structured calls, and it lives inside a loop that catches those calls and acts on them.

The first time I wired one of these loops up on a multi-model routing project, I assumed the hard part would be the model. It was not. The hard part was the referee. Most of the bugs I shipped lived in fifteen lines of dispatch code that I had written in a hurry because I assumed the model would do the heavy lifting.

There are two pieces worth naming carefully.

The first is the **tool schema**, a contract describing the tools that exist. Each tool has a name (a string), a description (what the tool does, written for the model to read), and a parameter spec (which arguments the tool takes, of which types). The schema is shown to the model in its system prompt or its training data. It tells the model what verbs it can use. Without the schema, the model has no idea what tools exist or how to call them, and it will either ignore tools entirely or invent calls to functions that do not exist.

The second is the **agent loop**, the runtime cycle that turns a single tool call into a multi-step interaction. The canonical version of this loop is called **ReAct**, short for Reason and Act, from a 2023 paper by Yao and colleagues. ReAct frames each step as three pieces: a **Thought** (the model's reasoning, written as plain text), an **Action** (the structured tool call, written as JSON), and an **Observation** (the result returned by the runtime after dispatching the action). The model emits Thought + Action. The runtime executes the action, captures the result, and inserts it back into the conversation as an Observation. The model reads the new context and emits the next Thought + Action. Repeat until the model emits a final answer instead of an action.

The shorthand for the loop is: think, act, observe, repeat. Three verbs and a cycle. That is the entire agent primitive that every modern tool-using language model is built on.

I drew this loop on a whiteboard at least a dozen times before I trusted it. The drawing always looked too simple to be doing real work. It is.

A small clarification, because the vocabulary is overloaded. "Function calling" and "tool use" are usually the same thing wearing different marketing hats. OpenAI's API calls it function calling. Anthropic's API calls it tool use. The mechanism is identical: a structured schema, a JSON call, a runtime that dispatches, a returned observation. The phrase "agent" gets used too, and it adds one more concept on top. An agent is usually a tool-using model running inside a loop with some goal beyond a single turn. The loop in this chapter is the agent primitive. Build it once, and the larger agent systems become recognizable.

One more naming convention worth pinning down. The verb that turns a parsed JSON call into an actual function invocation is called **dispatch**. The runtime looks at the name field, finds the matching function in its registry, calls it with the parameters from the JSON, and captures the return value. Dispatch is the moment where text becomes execution. Everything before dispatch is the model talking. Everything after is the runtime acting on what it heard.

That sentence sounds harmless. It is not. The boundary between "model talking" and "runtime acting" is exactly where most of the production bugs in tool use live, because the model produces text that looks like a valid call and the runtime trusts the text. If the model invents a tool name that does not exist, the runtime has to catch it. If the model emits a string where the schema expects an integer, the runtime has to catch it. The dispatch step is the only layer that has both a schema and a Python exception system. The model has neither.

## Why It Matters

Without tool use, a language model is a sealed room. It knows whatever was in its training data, and it cannot learn anything new during a conversation. Ask it the current price of a stock and it will guess. Ask it to compute 8347 times 91256 and it will be wrong, because token-by-token arithmetic is not what the model is for. Ask it to read a file on your laptop and it has no path to that file. The model's knowledge is frozen at training time, and its capabilities are bounded by what it can compute inside a forward pass.

With tool use, the room has windows. The model still cannot itself read a file or call an API, but it can ask the runtime to do those things on its behalf. The model becomes the planner of a system whose hands are made of other people's code. Python functions, HTTP clients, database drivers, shell commands. The shape of what a language model can do widens by an order of magnitude, and most of the widening comes from a few hundred lines of glue code wrapped around the model.

That asymmetry is the part that took me longest to internalize. A model adds an unbounded body of capabilities not by getting smarter, but by getting permission to ask other software to act on its behalf.

There is a second reason this chapter sits where it does. **Project 28: Retrieval-Augmented Generation** is going to build a tool-using model whose main tool is a vector search call. Retrieval is a special case of tool use. The model calls a `search(query: str)` function, the runtime executes the search, the runtime returns the top-k documents as the observation, and the model uses those documents in its next response. If you understand the loop in this chapter, retrieval becomes a single new tool added to the registry instead of a new mechanism. The chapter after will not re-explain ReAct. It will reach back to this chapter and say "the same loop, with one tool that happens to be a vector store."

There is also a third reason that becomes obvious the first time you ship a tool-using model into production. The model fails. The tool fails. The JSON is malformed. The arguments are the wrong type. The function raises an exception. None of these failures crash the program. They all become text inside the conversation. The model receives the broken text and tries to recover, and the recovery is sometimes wrong in expensive ways. Knowing what the loop looks like is the difference between a system you can debug and a system that produces bizarre outputs you cannot trace back to a cause.

On an edge deployment I worked on for community health workers, running a small model on Android against a structured protocol engine, we had a hard constraint that the structured output had to match the protocol schema exactly. There was no human in the loop who could catch a wrong field. The first version of that schema-enforcement layer was three lines of regex. The version we shipped was about four hundred lines of validation, with explicit named errors for every kind of malformed call. Almost none of those four hundred lines existed because we predicted the failure. They existed because we watched the failure happen and added a check after the fact.

## How to run this project

```bash
# Proxy run (tiny model, runs on CPU in <60s):
python projects/26_tool-use-and-function-calling/build.py --tiny

# Full lab (requires hardware — see setup/03_gpu-and-hardware-tiers.md):
python projects/26_tool-use-and-function-calling/build.py --full

# The BREAK IT experiment:
python projects/26_tool-use-and-function-calling/break_it.py
```

## Outputs

_To be captured in PR 3. Will include loss curves, sample generations, and any benchmark results._

## Read in the book

This project is Chapter 26 of *Under the Hood: Build Every Layer of a Large Language Model from Scratch*. Buy the book at <https://leanpub.com/under-the-hood>.
