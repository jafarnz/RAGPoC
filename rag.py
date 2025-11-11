import json, numpy as np, faiss, google.generativeai as genai

system_instruction = """
You are an autonomous category-lookup agent for a cosmetics brand.

Rules:not 
- Never ask follow-up questions or confirmations.
- Always treat the user's message as a final command.
- Always call find_best_category(query) to resolve vague or partial names like “UV”, “brows”, “hydrating”.
- Return the most relevant category identifier, even for high-level parents such as “Eyes” when no sub-category is specified.
- If the query clearly targets a specific child (e.g., “eye lines”, “brow makeup”), prefer that child over the parent.
- Respond with a short sentence that states the path and the identifier, e.g.:
    - "Skincare > By Category > UV Protection maps to ID 3fa9b1c2."
    - "Makeup > Eyes covers Mascaras, Eyeshadows, Eye Liners, Eyebrow Makeup; parent ID 07871a24."

"""

genai.configure(api_key="YOUR_API_KEY_HERE")
EMBED_MODEL = "text-embedding-004"
CHAT_MODEL = "gemini-2.5-flash"

with open("categories.json") as f:
    raw_categories = json.load(f)


def _normalise_variants(label):
    """Generate lowercase variants for approximate lookup."""

    base = label.lower()
    variants = {
        base,
        base.replace("&", "and"),
        base.replace("&", ""),
        base.replace("/", " "),
        base.replace("-", " "),
    }
    # Normalise spaces after replacements to keep comparisons tidy.
    variants = {" ".join(v.split()) for v in variants if v}
    compact = base.replace(" ", "")
    variants.add(compact)
    return variants, compact


def flatten_categories(tree):
    """Walk the nested category tree and emit flat entries with path metadata."""

    entries = []

    def walk(node, path, parent_path):
        meta = node.get("_meta", {})
        child_keys = [k for k in node.keys() if k != "_meta"]

        components = [segment.strip() for segment in path.split(" > ")]
        leaf = components[-1]
        leaf_variants, leaf_compact = _normalise_variants(leaf)
        path_variants, _ = _normalise_variants(path)
        term_variants = set()
        for segment in components:
            segment_variants, _ = _normalise_variants(segment)
            term_variants.update(segment_variants)
        term_variants.update(path_variants)

        entry = {
            "path": path,
            "id": meta.get("id"),
            "definition": meta.get("definition", ""),
            "parent": parent_path,
            "children": [f"{path} > {child}" for child in child_keys],
            "_leaf": leaf.lower(),
            "_leaf_compact": leaf_compact,
            "_search_terms": term_variants,
        }
        entries.append(entry)

        for child in child_keys:
            child_path = f"{path} > {child}"
            walk(node[child], child_path, path)

    for root_name, root_node in tree.items():
        walk(root_node, root_name, None)

    return entries


category_entries = flatten_categories(raw_categories)

path_to_entry = {entry["path"]: entry for entry in category_entries}

def describe_entry(entry):
    children = [child.split(" > ")[-1] for child in entry.get("children", [])]
    child_text = f"children: {', '.join(children)}" if children else "no direct children"
    parent_label = entry["parent"].split(" > ")[-1] if entry.get("parent") else "root"
    return f"{entry['path']} | parent: {parent_label} | {child_text} | {entry['definition']} | identifier {entry['id']}"

category_docs = [describe_entry(entry) for entry in category_entries]


print(" Building embeddings…")
embeds = [genai.embed_content(model=EMBED_MODEL, content=c)["embedding"] for c in category_docs]
index = faiss.IndexFlatL2(len(embeds[0]))
index.add(np.array(embeds).astype("float32"))
print(f"Indexed {len(category_entries)} categories\n")

def find_best_category(query):
    query = (query or "").strip()
    if not query:
        raise ValueError("query must be a non-empty string")

    q_norm = " ".join(query.lower().split())
    q_compact = q_norm.replace(" ", "")

    lexical_best = None
    lexical_score = -1

    for entry in category_entries:
        leaf = entry.get("_leaf", "")
        leaf_compact = entry.get("_leaf_compact", "")
        terms = entry.get("_search_terms", set())

        score = 0
        if not terms:
            terms = set()

        if q_norm == leaf:
            score = 6
        elif q_compact and q_compact == leaf_compact:
            score = 5
        elif q_norm in terms:
            score = 5
        elif leaf.startswith(q_norm):
            score = 4
        elif q_compact and leaf_compact.startswith(q_compact):
            score = 3
        elif q_norm and q_norm in leaf:
            score = 2
        elif q_compact and q_compact in leaf_compact:
            score = 2
        elif any(q_norm in term for term in terms if term):
            score = 1

        # Prefer deeper paths when scores tie to resolve parent vs child ties.
        depth = entry["path"].count(" > ")
        best_depth = lexical_best["path"].count(" > ") if lexical_best else -1

        if score > lexical_score or (score == lexical_score and depth > best_depth):
            lexical_best = entry
            lexical_score = score

    if lexical_best and lexical_score >= 2:
        match_entry = lexical_best
    else:
        q_emb = genai.embed_content(model=EMBED_MODEL, content=query)["embedding"]
        D, I = index.search(np.array([q_emb]).astype("float32"), 1)
        match_entry = category_entries[I[0][0]]

    return {
        "path": match_entry["path"],
        "id": match_entry["id"],
        "definition": match_entry["definition"],
        "children": [child.split(" > ")[-1] for child in match_entry.get("children", [])]
    }


model = genai.GenerativeModel(
    model_name=CHAT_MODEL,
    tools=[find_best_category],
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
