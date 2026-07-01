# HumorGen website

Static marketing site for HumorGen — no build step required.

## Local preview

```bash
cd website
python3 -m http.server 8000
# open http://localhost:8000
```

## Deploy to Cloudflare Pages (`humorgen.pages.dev`)

1. In [Cloudflare Dashboard](https://dash.cloudflare.com/) → **Workers & Pages** → **Create** → **Pages** → **Connect to Git**
2. Select repository: [edwardajayi/HumorGen](https://github.com/edwardajayi/HumorGen)
3. Build settings:
   - **Production branch:** `master`
   - **Framework preset:** None
   - **Build command:** *(leave empty)*
   - **Build output directory:** `website`
4. Deploy. Then add custom domain **humorgen.pages.dev** (or your `*.pages.dev` subdomain) under **Custom domains**.

## Files

- `index.html` — page structure and content
- `styles.css` — Anthropic-inspired warm theme (light + dark)
- `script.js` — theme toggle, scroll reveal, persona ticker
- `humorgen-logo.png` — brand logo
- `humorgen.md` — source content reference
