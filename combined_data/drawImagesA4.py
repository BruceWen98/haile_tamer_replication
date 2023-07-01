from PIL import Image
import os
from PyPDF2 import PdfMerger

def image_to_pdf(images_folder, pdf_file, images_per_row=3):
    images = []
    for image in os.listdir(images_folder):
        img_path = os.path.join(images_folder, image)
        if img_path.endswith(".png"):
            images.append(Image.open(img_path))

    rows = len(images) // images_per_row + (len(images) % images_per_row > 0)
    max_width = max([img.width for img in images])
    max_height = max([img.height for img in images])
    total_width = max_width * images_per_row
    total_height = max_height * rows
    output_image = Image.new("RGB", (total_width, total_height))

    x_offset = 0
    y_offset = 0
    for img in images:
        output_image.paste(img, (x_offset, y_offset))
        x_offset += max_width
        if x_offset >= total_width:
            x_offset = 0
            y_offset += max_height

    output_image.save(pdf_file, "pdf")


def get_folders_in_directory(parent_folder):
    return [f.path for f in os.scandir(parent_folder) if f.is_dir()]

def mergePDFs(pdfList, output):
    merger = PdfMerger()
    for pdf in pdfList:
        merger.append(pdf)
    merger.write(output)
    merger.close()
    return


img_folder = "/Users/brucewen/Desktop/honors_thesis/estimation/combined_data/simulationArtImp20/ALL_d_art_imp20"
pdf_file = "/Users/brucewen/Desktop/honors_thesis/estimation/combined_data/simulationArtImp20/varyingN.pdf"
image_to_pdf(img_folder, pdf_file, 3)

# # RUN CODE.
# images_folders = get_folders_in_directory("/Users/brucewen/Desktop/honors_thesis/estimation/combined_data/categorized_data_results/")
# pdf_file = "/Users/brucewen/Desktop/honors_thesis/estimation/combined_data/pdf_results/testing.pdf"

# for i,f in enumerate(images_folders):
#     image_to_pdf(f, pdf_file.split(".")[0] + "_{}.pdf".format(i), 3)

# pdf_files = ["/Users/brucewen/Desktop/honors_thesis/estimation/combined_data/pdf_results/" + f for f in os.listdir(pdf_file.rsplit("/",1)[0]+"/") if f.endswith(".pdf")]
# print(pdf_files)
# mergePDFs(pdf_files, "/Users/brucewen/Desktop/honors_thesis/estimation/combined_data/pdf_results/FINAL.pdf")