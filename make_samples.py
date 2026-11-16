"""Gera os PDFs de exemplo usados na demonstração.

Uso: python make_samples.py
Saída: inputs/pdfs/artigo_rag.pdf e inputs/pdfs/artigo_ia_educacao.pdf
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

ARTIGO_RAG = {
    "title": "Recuperação Aumentada por Geração (RAG) na revisão sistemática de literatura: um estudo de caso",
    "authors": "Ana Beatriz Ferreira, Carlos Eduardo Lima — Universidade Federal de Santa Catarina (UFSC)",
    "sections": [
        (
            "Resumo",
            [
                "A revisão sistemática de literatura exige que o pesquisador leia, classifique e relacione dezenas "
                "de artigos científicos. Este trabalho investiga se a técnica de Recuperação Aumentada por Geração "
                "(RAG, do inglês Retrieval-Augmented Generation) reduz o esforço de triagem e melhora a rastreabilidade "
                "das informações extraídas. Foi conduzido um estudo de caso com um corpus de 42 artigos em PDF sobre "
                "engenharia de software, totalizando 517 páginas. Os resultados indicam redução de 63% no tempo de "
                "triagem inicial e precisão de 0,82 nos cinco trechos mais relevantes recuperados por pergunta.",
            ],
        ),
        (
            "1. Introdução",
            [
                "O volume de publicações científicas cresce em ritmo superior à capacidade de leitura individual. "
                "Estudantes de pós-graduação relatam dificuldade não apenas em localizar artigos, mas em correlacionar "
                "ideias entre textos com vocabulários distintos. Ferramentas de busca tradicionais recuperam documentos "
                "inteiros, e não as passagens que respondem a uma pergunta específica.",
                "Modelos de linguagem de grande porte (LLMs) poderiam sintetizar esse conteúdo, porém apresentam dois "
                "problemas conhecidos: alucinação e ausência de rastreabilidade. A arquitetura RAG endereça ambos ao "
                "condicionar a geração a trechos recuperados de uma base documental, permitindo que cada afirmação seja "
                "associada ao arquivo e à página de origem.",
            ],
        ),
        (
            "2. Fundamentação teórica",
            [
                "RAG combina três componentes: indexação, recuperação e geração. Na indexação, cada página do PDF é "
                "dividida em trechos (chunks) de aproximadamente 900 caracteres, com sobreposição de 150 caracteres para "
                "preservar o contexto nas fronteiras. Cada trecho é convertido em um vetor denso por um modelo de "
                "embeddings, como o text-embedding-3-small, e armazenado em um índice vetorial.",
                "A recuperação pode ser densa, lexical ou híbrida. A busca densa calcula similaridade de cosseno entre a "
                "pergunta e os vetores dos trechos. A busca lexical BM25 pontua termos exatos, sendo superior em nomes "
                "próprios, siglas e números. A fusão por Reciprocal Rank Fusion (RRF) combina ambos os rankings sem exigir "
                "calibração de escalas, usando a constante k igual a 60.",
                "Na etapa de geração, os trechos recuperados são numerados e inseridos no prompt. O modelo é instruído a "
                "responder somente com base nesse contexto e a citar a fonte de cada afirmação no formato [n].",
            ],
        ),
        (
            "3. Método",
            [
                "O corpus foi composto por 42 artigos publicados entre 2019 e 2025, obtidos nas bases Scopus e IEEE "
                "Xplore. Todos os arquivos estavam em formato PDF com camada de texto. Foram descartados três artigos "
                "escaneados sem OCR, pois a extração de texto retornava páginas vazias.",
                "A configuração experimental utilizou chunks de 900 caracteres, sobreposição de 150, cinco trechos por "
                "pergunta (top-k igual a 5) e temperatura 0,2. Foram formuladas 30 perguntas de revisão, divididas em "
                "factuais (números e definições) e relacionais (comparações entre artigos).",
                "A avaliação considerou três métricas: tempo de triagem, precisão do top-5 e taxa de afirmações sem "
                "respaldo no documento (alucinação). Dois avaliadores independentes julgaram as respostas com apoio dos "
                "trechos citados.",
            ],
        ),
        (
            "4. Resultados",
            [
                "O tempo médio de triagem caiu de 74 minutos para 27 minutos por artigo, uma redução de 63%. A precisão "
                "do top-5 foi de 0,82 para perguntas factuais e 0,71 para perguntas relacionais. A taxa de alucinação "
                "foi de 18% no baseline sem recuperação e de 4% no sistema com RAG e citação obrigatória.",
                "Os erros remanescentes concentraram-se em tabelas e figuras: a extração de texto de PDFs frequentemente "
                "quebra a ordem de leitura desses elementos. Perguntas que exigiam soma de valores presentes em tabelas "
                "obtiveram o pior desempenho, com precisão de 0,48.",
            ],
        ),
        (
            "5. Limitações",
            [
                "A principal limitação é a dependência da qualidade da camada de texto do PDF. Documentos escaneados "
                "exigem OCR antes da indexação. Além disso, o chunking por caracteres pode separar uma tabela de seu "
                "enunciado, e perguntas que exigem agregação de muitos trechos ainda apresentam erros.",
                "O estudo de caso foi restrito a uma área do conhecimento e a um único modelo de embeddings, o que "
                "limita a generalização dos resultados.",
            ],
        ),
        (
            "6. Conclusão",
            [
                "A arquitetura RAG mostrou-se viável para apoiar a revisão de literatura, reduzindo tempo de triagem e "
                "aumentando a rastreabilidade das afirmações. Trabalhos futuros devem investigar chunking semântico, "
                "extração estruturada de tabelas e avaliação com revisores humanos em maior escala.",
            ],
        ),
    ],
}

ARTIGO_EDUCACAO = {
    "title": "IA generativa no apoio à escrita acadêmica: um estudo com estudantes de Engenharia de Software",
    "authors": "Mariana Souza, Rafael Antunes — Instituto de Computação, Universidade Estadual de Campinas (UNICAMP)",
    "sections": [
        (
            "Resumo",
            [
                "Este artigo apresenta um estudo exploratório sobre o uso de assistentes de IA generativa no apoio à "
                "escrita acadêmica. Participaram 57 estudantes de Engenharia de Software, distribuídos em um grupo de "
                "uso assistido e um grupo de controle, ao longo de três semanas. O grupo assistido apresentou ganho de "
                "27% em clareza textual segundo avaliação cega de dois professores. Contudo, 41% dos participantes "
                "relataram dificuldade em verificar as fontes sugeridas pela ferramenta.",
            ],
        ),
        (
            "1. Introdução",
            [
                "A escrita acadêmica exige argumentação fundamentada e diálogo com a literatura. Estudantes de graduação "
                "frequentemente dominam o conteúdo técnico, mas têm dificuldade em organizar a argumentação e citar "
                "adequadamente as fontes. Assistentes baseados em modelos de linguagem podem apoiar essas etapas, mas "
                "levantam questões éticas relacionadas a autoria, plágio e confiabilidade das referências.",
                "Este estudo busca responder a duas perguntas: (i) assistentes de IA melhoram a clareza e a estrutura de "
                "textos acadêmicos? (ii) quais riscos os estudantes percebem ao utilizá-los?",
            ],
        ),
        (
            "2. Metodologia",
            [
                "O experimento adotou abordagem quasi-experimental com pré-teste e pós-teste. Os 57 participantes eram "
                "estudantes do quinto semestre, com idade média de 22 anos. O grupo assistido utilizou um assistente "
                "configurado para responder apenas com base em uma pasta de artigos fornecida pelos pesquisadores, "
                "seguindo a abordagem de Recuperação Aumentada por Geração.",
                "A avaliação cega considerou quatro critérios em escala de 1 a 5: clareza, coesão, adequação das citações "
                "e originalidade. Complementarmente, foram realizadas oito entrevistas semiestruturadas para compreender "
                "as percepções dos estudantes.",
            ],
        ),
        (
            "3. Resultados",
            [
                "O ganho médio em clareza foi de 27%, e em coesão, de 19%. A adequação das citações não apresentou "
                "diferença estatisticamente significativa entre os grupos, com média de 3,4 no grupo assistido e 3,3 no "
                "controle. A originalidade foi ligeiramente menor no grupo assistido, sugerindo risco de padronização "
                "de estilo.",
                "Nas entrevistas, 41% dos estudantes afirmaram ter dificuldade em verificar se as referências sugeridas "
                "existiam de fato. Outros relatos mencionaram risco de dependência da ferramenta e necessidade de "
                "declarar o uso de IA nos trabalhos.",
            ],
        ),
        (
            "4. Discussão e recomendações",
            [
                "Os resultados sugerem que assistentes ancorados em documentos podem apoiar a escrita sem substituir a "
                "autoria. A citação obrigatória de trechos [n], com arquivo e página, foi apontada como o recurso mais "
                "útil para conferência das fontes.",
                "Recomenda-se que instituições definam políticas explícitas de uso de IA, incluindo declaração de uso, "
                "revisão humana das citações e cuidado com dados pessoais, em conformidade com a Lei Geral de Proteção "
                "de Dados (LGPD).",
            ],
        ),
        (
            "5. Conclusão",
            [
                "Assistentes de IA generativa com recuperação documental melhoraram a clareza e a coesão dos textos, "
                "mas não eliminaram a necessidade de verificação humana. A transparência das citações é condição para "
                "uso responsável na educação superior.",
            ],
        ),
    ],
}


def build_pdf(path: Path, article: dict) -> None:
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TituloArtigo", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=16, leading=20, spaceAfter=10
    )
    author_style = ParagraphStyle(
        "Autores", parent=styles["Normal"], fontSize=9, leading=12, textColor="#555555", spaceAfter=18
    )
    heading_style = ParagraphStyle(
        "Secao", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12, leading=15,
        spaceBefore=12, spaceAfter=6, textColor="#8a4b08",
    )
    body_style = ParagraphStyle("Corpo", parent=styles["BodyText"], fontSize=10.5, leading=15, spaceAfter=7,
                                alignment=4)

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor("#999999")
        canvas.drawString(2 * cm, 1.2 * cm, "Documento de exemplo gerado para o desafio PDF Chat · DIO")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"p. {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(path), pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm, leftMargin=2.2 * cm, rightMargin=2.2 * cm,
        title=article["title"], author=article["authors"],
    )
    story = [Paragraph(article["title"], title_style), Paragraph(article["authors"], author_style)]
    for number, (heading, paragraphs) in enumerate(article["sections"]):
        if number == 2:  # quebra de página para gerar páginas distintas no índice
            story.append(PageBreak())
        story.append(Paragraph(heading, heading_style))
        for paragraph in paragraphs:
            story.append(Paragraph(paragraph.replace("&", "&amp;"), body_style))
        story.append(Spacer(1, 4))
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main() -> None:
    out = Path(__file__).resolve().parent / "inputs" / "pdfs"
    out.mkdir(parents=True, exist_ok=True)
    for name, article in (("artigo_rag.pdf", ARTIGO_RAG), ("artigo_ia_educacao.pdf", ARTIGO_EDUCACAO)):
        build_pdf(out / name, article)
        print(f"criado: {out / name}")
    print("\nAgora rode: python ingest.py inputs --rebuild")


if __name__ == "__main__":
    main()
