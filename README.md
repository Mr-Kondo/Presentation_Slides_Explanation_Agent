# Presentation Slides (Metsuke-Core)

## PDF build with Mermaid (Pandoc)

This project uses Pandoc with a Lua filter to render Mermaid diagrams into SVG and embed them in the PDF.

### Prerequisites
- Pandoc
- Node.js + mermaid-cli (`mmdc`)
- wkhtmltopdf (Pandoc PDF engine)

### Build command
```
pandoc slides.md \
	--from=markdown+raw_html \
	--to=html \
	--standalone \
	--lua-filter=mermaid-filter.lua \
	--pdf-engine=wkhtmltopdf \
	--pdf-engine-opt=--enable-local-file-access \
	-o slides.pdf
```

### Notes
- Mermaid blocks are rendered by `mmdc` via [mermaid-filter.lua](mermaid-filter.lua).
- If `mmdc` is not on PATH, set the `MMDC` environment variable to its absolute path.
