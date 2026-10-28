"""Render the actual PDF, make a contact sheet, and check saved arrays/artifacts."""
from pathlib import Path
import argparse, json, math, subprocess
import numpy as np
from PIL import Image, ImageDraw
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"outputs"

def main(final=False):
    with np.load(OUT/"fits.npz") as z:
        invalid=[k for k in z.files if z[k].dtype.kind in "fc" and not np.isfinite(z[k]).all()]
        assert not invalid, invalid
        assert len(z["t"])==7451 and len(z["bands"])==17
        assert z["q_baseline"].shape==(50,250)
        n_maps=sum(k.startswith("q_") for k in z.files)
    pdf=OUT/"Mrk817_reproduction.pdf"
    reader=PdfReader(pdf)
    page_count=len(reader.pages)
    if final: assert page_count==19, page_count
    qc=OUT/"qc"
    qc.mkdir(exist_ok=True)
    subprocess.run(["pdftoppm","-png","-scale-to","900",str(pdf),str(qc/"page")],check=True)
    pages=sorted(qc.glob("page-*.png"))
    # Only the pages in this PDF; earlier renders can have more pages.
    pages=pages[:page_count]
    columns=4; tw,th=360,265
    sheet=Image.new("RGB",(columns*tw,math.ceil(page_count/columns)*th),"#dddddd")
    draw=ImageDraw.Draw(sheet)
    for i,path in enumerate(pages):
        with Image.open(path) as im:
            im=im.convert("RGB")
            im.thumbnail((tw-12,th-28))
            x=(i%columns)*tw+(tw-im.width)//2
            y=(i//columns)*th+23
            sheet.paste(im,(x,y))
        draw.text(((i%columns)*tw+10,(i//columns)*th+5),f"PDF page {i+1}",fill="black")
    sheet.save(qc/"contact_sheet.jpg",quality=88)
    figures=[]
    for path in sorted((OUT/"figures").glob("*.png")):
        with Image.open(path) as im:
            im.verify()
        figures.append({"file":path.name,"bytes":path.stat().st_size})
    result={"array_checks":"passed","n_data":7451,"n_bands":17,"map_shape":[50,250],
            "n_saved_maps":n_maps,"pdf_pages":page_count,"pdf_bytes":pdf.stat().st_size,
            "pdf_render":"passed","figures":figures,"final_atlas_expected_pages":19,
            "atlas_complete":page_count==19,"contact_sheet":str(qc/"contact_sheet.jpg")}
    if (OUT/"lag_summary.json").exists():
        lag=json.loads((OUT/"lag_summary.json").read_text())
        result["lag_50tau_checks"]={k:v["converged_50tau"] for k,v in lag["cases"].items()}
    (OUT/"qa.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--final",action="store_true")
    main(parser.parse_args().final)
