# HumorGen website — Google Analytics & Search Console setup

Free monitoring for **humorgen.pages.dev** (or your custom domain) on Cloudflare Pages.

Per [Google's guide](https://developers.google.com/search/docs/monitor-debug/google-analytics-search-console):

- **Search Console** — what happens **before** someone lands on your site (Google Search impressions, clicks, queries)
- **Google Analytics** — what happens **after** they arrive (pages visited, time on site, traffic sources)

Both tools are **100% free** for a site like HumorGen. No credit card required.

| Tool | Cost |
|------|------|
| Google Search Console | Free |
| Google Analytics 4 (GA4) | Free |
| Looker Studio (optional combined dashboard) | Free |

Paid tiers (GA4 360, BigQuery at scale) are enterprise-only — you don't need them.

---

## Step 1 — Create Google Analytics 4

1. Go to [analytics.google.com](https://analytics.google.com)
2. **Admin** (gear, bottom-left) → **Create** → **Property**
3. Name it `HumorGen`, set timezone
4. **Data streams** → **Add stream** → **Web**
5. Enter your site URL: `https://humorgen.pages.dev` (or your custom domain)
6. Copy your **Measurement ID** — looks like `G-XXXXXXXXXX`

---

## Step 2 — Add the tag to the website

Add this inside `<head>` in `index.html`, **before** `</head>`:

```html
<!-- Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-XXXXXXXXXX"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-XXXXXXXXXX');
</script>
```

Replace `G-XXXXXXXXXX` with your real Measurement ID.

Deploy to Cloudflare Pages, visit the site once, then check **Analytics → Reports → Realtime** — you should see yourself.

---

## Step 3 — Set up Search Console

1. Go to [search.google.com/search-console](https://search.google.com/search-console)
2. **Add property** → **URL prefix**
3. Enter `https://humorgen.pages.dev` (or your custom domain)

### Verify ownership

**Option A — HTML meta tag (recommended for static sites)**

Search Console gives you a tag like:

```html
<meta name="google-site-verification" content="abc123..." />
```

Add it to `<head>` in `index.html`, deploy, then click **Verify** in Search Console.

**Option B — DNS (custom domain on Cloudflare)**

Add the TXT record Search Console provides in Cloudflare DNS → **Verify**.

---

## Step 4 — Add sitemap and robots.txt

Create `website/sitemap.xml`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://humorgen.pages.dev/</loc>
    <lastmod>2026-07-02</lastmod>
    <changefreq>monthly</changefreq>
    <priority>1.0</priority>
  </url>
</urlset>
```

Create `website/robots.txt`:

```
User-agent: *
Allow: /

Sitemap: https://humorgen.pages.dev/sitemap.xml
```

Update URLs if you use a custom domain.

In Search Console: **Sitemaps** → submit `https://humorgen.pages.dev/sitemap.xml`

---

## Step 5 — Link Search Console to Analytics

This surfaces Search queries inside Analytics.

1. **Search Console** → **Settings** → **Associations** → **Associate** with your GA4 property

   **or**

2. **Analytics** → **Admin** → **Product links** → **Search Console links** → **Link**

After linking, see **Reports → Search Console** in Analytics (queries, landing pages from Google organic search).

---

## Step 6 — Optional: combined dashboard in Looker Studio

Google provides a [Looker Studio template](https://developers.google.com/search/docs/monitor-debug/google-analytics-search-console) that puts Search Console (blue) and Analytics (orange) side by side.

1. Open the template from Google's doc
2. Click **Use my own data**
3. Connect **Search Console** → choose **URL Impression**
4. Connect **Google Analytics**
5. Wire each chart to your data sources

| Metric | Tool | What it tells you |
|--------|------|-------------------|
| Clicks | Search Console | People clicked your link in Google results |
| CTR | Search Console | How often searchers click when they see you |
| Sessions | Analytics | Visits from organic Google search |
| Engagement rate | Analytics | Whether people actually read the page |

Clicks and sessions won't match exactly — that's normal ([Google explains why](https://developers.google.com/search/docs/monitor-debug/google-analytics-search-console#understanding-data-discrepancies-between-google-analytics-and-search-console)).

---

## What to expect for HumorGen

- **Indexing takes time** — a new research site may take days to weeks before Search Console shows meaningful data
- **Early traffic** is often direct/referral (Hugging Face, arXiv, GitHub) — Analytics shows this under **Acquisition**
- **Search queries** like "HumorGen", "computational humor", "Humor Transfer Bench" may appear once indexed

---

## Checklist

| Step | Done? |
|------|-------|
| Create GA4 property + get `G-XXXXXXXXXX` | ☐ |
| Add gtag to `index.html` `<head>` | ☐ |
| Deploy to Cloudflare Pages | ☐ |
| Verify in Search Console (meta tag or DNS) | ☐ |
| Add `sitemap.xml` + `robots.txt` | ☐ |
| Submit sitemap in Search Console | ☐ |
| Link Search Console ↔ Analytics | ☐ |

---

## References

- [Using Search Console and Google Analytics data for SEO](https://developers.google.com/search/docs/monitor-debug/google-analytics-search-console)
- [Get started with Search Console](https://support.google.com/webmasters/answer/9128669)
- [Set up Google Analytics](https://support.google.com/analytics/answer/9304153)
- [Cloudflare Pages deploy](./README.md)
