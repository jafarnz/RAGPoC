import json, re, numpy as np, faiss, google.generativeai as genai

system_instruction = """
You are an autonomous category-management agent for a cosmetics brand.

Rules:
- Never ask follow-up questions or confirmations.
- Always treat the user's message as a final command.
- Use find_best_category(query) to resolve vague or partial names like “UV”, “brows”, “hydrating”.
- If a number is mentioned, assume it's the quantity to set.
- If verbs like 'update', 'set', 'change', or 'edit' appear, call set_qty(category, number) immediately.
- If verbs like 'show', 'view', 'see', 'stock', or 'qty' appear, call get_qty(category). 
- For all queries, if a well-constructed and instruction set is defined in a sentence, even if the verbs do not appear, follow through with the best possible action.
- Prefer categories ending with 'Protection' over brand lines like 'UV Expert' when both match 'UV'.
- Reply only with a short, natural sentence describing what you did, e.g.:
    - "Updated Skincare > By Category > UV Protection to 100 units."
    - "Skincare > By Category > Serum currently has 42 in stock."

"""

genai.configure(api_key="API_KEY")
EMBED_MODEL = "text-embedding-004"
CHAT_MODEL = "gemini-2.5-flash"

with open("categories.json") as f:
    data = json.load(f)
with open("category_data.json") as f:
    cat_data = json.load(f)

def flatten_categories(d, parent=""):
    res = []
    for k, v in d.items():
        path = f"{parent} > {k}" if parent else k
        if isinstance(v, dict):
            res += flatten_categories(v, path)
        elif isinstance(v, list):
            for item in v:
                res.append(f"{path} > {item}")
    return res

categories = flatten_categories(data)


print(" Building embeddings…")
embeds = [genai.embed_content(model=EMBED_MODEL, content=c)["embedding"] for c in categories]
index = faiss.IndexFlatL2(len(embeds[0]))
index.add(np.array(embeds).astype("float32"))
print(f"Indexed {len(categories)} categories\n")

def find_best_category(query):
    q_emb = genai.embed_content(model=EMBED_MODEL, content=query)["embedding"]
    D, I = index.search(np.array([q_emb]).astype("float32"), 1)
    return categories[I[0][0]]



def get_qty(category):
    return cat_data.get(category, {}).get("qty", None)

def set_qty(category, new_qty):
    if category not in cat_data:
        cat_data[category] = {"qty": new_qty}
    else:
        cat_data[category]["qty"] = new_qty
    with open("category_data.json", "w") as f:
        json.dump(cat_data, f, indent=2)
    return new_qty


model = genai.GenerativeModel(
    model_name=CHAT_MODEL,
    tools=[find_best_category, get_qty, set_qty],
    system_instruction=system_instruction
)





def handle_response(resp):
    """Recursively handle Gemini responses and tool calls until we get text output."""
    if not resp.candidates or not resp.candidates[0].content.parts:
        return resp.text

    for part in resp.candidates[0].content.parts:
        if hasattr(part, "function_call") and part.function_call:
            fn_name = part.function_call.name
            args = dict(part.function_call.args)  
            print(f"Gemini called tool: {fn_name} {args}")

            
            if fn_name == "find_best_category":
                result = find_best_category(**args)
            elif fn_name == "get_qty":
                result = get_qty(**args)
            elif fn_name == "set_qty":
                result = set_qty(**args)
            else:
                result = f"[error] unknown tool {fn_name}"

           
            payload = result if isinstance(result, dict) else {"result": result}

            follow = chat.send_message({
                "function_response": {
                    "name": fn_name,
                    "response": payload
                }
            })
            return handle_response(follow)  

    
    return resp.text


if __name__ == "__main__":
    chat = model.start_chat(history=[])
    while True:
        q = input("\n Query: ").strip()
        if not q:
            print("  please type something.")
            continue

        response = chat.send_message(q)
        print(handle_response(response))
