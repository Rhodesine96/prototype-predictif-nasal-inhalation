from io import BytesIO
from html import escape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def pdf_report(title, lines):
    stream = BytesIO()
    doc = SimpleDocTemplate(stream)
    styles = getSampleStyleSheet()
    story = [Paragraph(escape(title), styles['Title']), Spacer(1,16)]
    for line in lines:
        story.extend([Paragraph(escape(str(line)),styles['BodyText']),Spacer(1,9)])
    doc.build(story)
    return stream.getvalue()
