from pathlib import Path


SAMPLE_TEXT = [
    "Intelligent Document Assistant Sample",
    "",
    "This sample document describes a production-style retrieval augmented generation system.",
    "The application lets users upload PDF files, extracts text, cleans the content, and splits the text into overlapping semantic chunks.",
    "Each chunk is embedded and stored in a persistent ChromaDB vector database so future documents can be added without rebuilding the whole index.",
    "",
    "When a user asks a question, the system retrieves the most relevant chunks, sends those chunks to the language model as context, and instructs the model to answer only from the uploaded documents.",
    "The interface displays source file names, page numbers, chunk text, retrieval scores, response time, and retrieval latency.",
    "",
    "The evaluation layer measures answer relevance, context relevance, faithfulness, hallucination risk, retrieval quality, and latency.",
    "These metrics are saved to local history and displayed in a dashboard so engineers can monitor RAG quality over time.",
]


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def create_pdf(path: Path, lines: list[str]) -> None:
    """Create a small valid PDF using only the Python standard library."""
    path.parent.mkdir(parents=True, exist_ok=True)

    text_commands = ["BT", "/F1 11 Tf", "72 750 Td", "14 TL"]
    for line in lines:
        text_commands.append(f"({_escape_pdf_text(line)}) Tj")
        text_commands.append("T*")
    text_commands.append("ET")
    stream = "\n".join(text_commands).encode("latin-1")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length "
        + str(len(stream)).encode("ascii")
        + b" >>\nstream\n"
        + stream
        + b"\nendstream",
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = []
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode("ascii"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode(
            "ascii"
        )
    )
    path.write_bytes(pdf)


if __name__ == "__main__":
    output_path = Path(__file__).resolve().parents[1] / "data" / "samples" / "rag_system_overview.pdf"
    create_pdf(output_path, SAMPLE_TEXT)
    print(f"Created {output_path}")
