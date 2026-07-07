# Deploy to Hugging Face Spaces (free, permanent)

Hugging Face Spaces free tier gives you a permanent public URL, no credit card, no expiry. A Space "sleeps" after ~48h without traffic and wakes automatically on the next visit (~1–2 min cold start while models load) — it is never deleted.

## One-time setup

1. **Groq API key** (free): sign up at https://console.groq.com → API Keys → Create. Copy the `gsk_...` key.

2. **Hugging Face account** (free): https://huggingface.co/join

3. **Create the Space**: https://huggingface.co/new-space
   - Space name: `self-healing-rag`
   - License: MIT
   - SDK: **Streamlit**
   - Hardware: **CPU basic (free)**
   - Visibility: **Public** (required for free tier + lets recruiters see it)

4. **Add your Groq key as a secret** (never commit it):
   Space page → Settings → Variables and secrets → New secret
   - Name: `GROQ_API_KEY`
   - Value: your `gsk_...` key

## Push the code

```bash
cd "/path/to/RAG Project"

# HF uses git. Get a write token at https://huggingface.co/settings/tokens (type: Write)
git init
git add .
git commit -m "Self-healing RAG"
git remote add space https://huggingface.co/spaces/<YOUR_USERNAME>/self-healing-rag
git push --force space main
```

When prompted for credentials: username = your HF username, password = the **write token** (not your account password).

The Space builds automatically (~5 min first time: installs deps, downloads models on first query). Your permanent URL:

```
https://huggingface.co/spaces/<YOUR_USERNAME>/self-healing-rag
```

## Updating later

```bash
git add . && git commit -m "update" && git push space main
```

## Notes

- The README.md frontmatter (`sdk: streamlit`, `app_file: app.py`) is what tells Spaces how to run the app — already set up.
- `requirements.txt` is installed automatically by Spaces.
- If no `GROQ_API_KEY` secret is set, the app falls back to asking visitors for their own key in the sidebar — the demo still works.
- Free CPU basic has 16 GB RAM; MiniLM + DeBERTa NLI use ~2 GB. Plenty.
- Groq free tier limits (as of mid-2026): ~30 requests/min — fine for a demo; each query uses 1–4 LLM calls depending on healing.
- To keep the Space from sleeping before an interview/demo, just open the URL once a few minutes beforehand.
