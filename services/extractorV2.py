from pix2text import Pix2Text

img_fp = "services/page1.pdf"
p2t = Pix2Text.from_config()
doc = p2t.recognize_pdf(img_fp, page_numbers=[0, 1])
doc.to_markdown('output-md')  # The exported Markdown information is saved in the output-md directory