"""Render the English public reports. Reads Markdown/aggregates, never datasets."""
import argparse
import html
import json
from pathlib import Path
import re
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                               TableStyle, PageBreak, Image, KeepTogether)

ROOT = Path(__file__).resolve().parents[1]
NAVY = colors.HexColor('#17324D')
ST = getSampleStyleSheet()
ST.add(ParagraphStyle('ReportTitle', fontName='Helvetica-Bold',fontSize=20,
                      leading=23,textColor=NAVY,spaceAfter=11))
ST.add(ParagraphStyle('Section',fontName='Helvetica-Bold',fontSize=12,
                      leading=15,textColor=NAVY,spaceBefore=10,spaceAfter=4,keepWithNext=True))
ST.add(ParagraphStyle('Text',fontName='Helvetica',fontSize=9.5,leading=12.5,spaceAfter=6))
ST.add(ParagraphStyle('OverviewTitle',fontName='Helvetica-Bold',fontSize=16,leading=19,textColor=NAVY,spaceAfter=8))
ST.add(ParagraphStyle('Cell',fontName='Helvetica',fontSize=8,leading=10.7,spaceAfter=0))
ST.add(ParagraphStyle('Small',fontName='Helvetica',fontSize=8,leading=11,spaceAfter=6))


def inline(text):
    text=text.replace('\u2011','-').replace('\u2013','-').replace('\u2014','-')
    text=html.escape(text)
    text=re.sub(r'\[([^]]+)\]\(([^)]+)\)',r'\1',text)
    text=re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',text)
    text=re.sub(r'`([^`]+)`',r'<font face="Courier">\1</font>',text)
    # Long hashes remain exact in the PDF's text but can wrap at the visual break.
    text=re.sub(r'([a-f0-9]{32})([a-f0-9]{32})',r'\1<br/>\2',text)
    return text


def markdown(body):
    story=[]; lines=body.splitlines();i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line: i+=1;continue
        if line.startswith('```'):
            i+=1;code=[]
            while i<len(lines) and not lines[i].startswith('```'):
                code.append(lines[i]);i+=1
            story.append(Paragraph('<font face="Courier">'+html.escape(' '.join(code))+'</font>',ST['Small']))
            i+=1;continue
        if line.startswith('|'):
            grid=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                cells=[x.strip() for x in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[: -]+',x) for x in cells):
                    grid.append([Paragraph(inline(x),ST['Cell']) for x in cells])
                i+=1
            n=len(grid[0]);w=491
            widths=([w*.37]+[w*.63/(n-1)]*(n-1)) if n>2 else [w*.46,w*.54]
            table=Table(grid,colWidths=widths,repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#EAF1F5')),
                ('LINEBELOW',(0,0),(-1,0),.8,NAVY),('VALIGN',(0,0),(-1,-1),'TOP'),
                ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
                ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
                ('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#D8E0E7'))]))
            story += [table,Spacer(1,8)];continue
        if line.startswith('# '):story.append(Paragraph(inline(line[2:]),ST['ReportTitle']))
        elif line.startswith('## '):
            if line=='## What mattered causally':story.append(PageBreak())
            story.append(Paragraph(inline(line[3:]),ST['Section']))
        elif line.startswith('### '):story.append(Paragraph(inline(line[4:]),ST['Section']))
        else:
            chunk=[line];i+=1
            while i<len(lines) and lines[i].strip() and not lines[i].lstrip().startswith(('#','|','- ')):
                chunk.append(lines[i].strip());i+=1
            para=' '.join(chunk)
            story.append(Paragraph(inline(para),ST['Small'] if 'SHA-256:' in para or para.startswith('Copyright') else ST['Text']))
            continue
        i+=1
    return story


def footer(canvas,doc):
    canvas.saveState();canvas.setStrokeColor(colors.HexColor('#D6E0E7'))
    canvas.line(52,43,543,43);canvas.setFont('Helvetica',8);canvas.setFillColor(NAVY)
    canvas.drawString(52,29,'Yacine Zahouani | ITA-ITAndroids | Public technical edition 1.0.0')
    canvas.drawRightString(543,29,str(doc.page));canvas.restoreState()


def build(path,story,title):
    doc=SimpleDocTemplate(str(path),pagesize=A4,leftMargin=52,rightMargin=52,
        topMargin=48,bottomMargin=57,title=title,author='Yacine Zahouani',pageCompression=1)
    doc.build(story,onFirstPage=footer,onLaterPages=footer)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    mapping=json.loads((ROOT/'reproducibility/enseirb-references.json').read_text())['references']
    for row in mapping:
        src=ROOT/row['path'];body=src.read_text(encoding='utf-8')
        build(a.output/(src.stem+'.pdf'),markdown(body),body.splitlines()[0][2:])
    # Public overview, distinct from the administrative ENSEIRB submission.
    sections=[
      ('Humanoid perception under CPU constraints','system-overview',
       'A research-engineering programme by Yacine Zahouani, ITA-ITAndroids, supervised by Marcos Maximo. '
       'The inherited robot and software provide the system context. Personal contributions cover data quality, '
       'neural architecture studies, CPU export and measurement, deterministic replay and probabilistic debugging. '
       'This public overview accompanies sixteen source-linked technical reports. It is not an official school submission.', '6-21'),
      ('Reliable labels and useful segmentation','segmentation-tradeoff',
       'Trusted masks, manual line correction and deterministic evaluation made comparisons meaningful. '
       'Two historical splits contained 54 and 66 overlapping sequence groups, not a 54/66 ratio. '
       'The explicit-line reference reached line Dice 0.848773 on corrected final-only evidence. '
       'Binary MobileNet and Fast-SCNN occupy different quality/CPU points on an open development bridge. '
       'Their detector outputs are constant compatibility placeholders. No independent Chape accuracy is claimed.', '6-9'),
      ('Controlled detection and real target timing','detector-comparison',
       'The same-data study used 12,018 images and three seeds per 640-pixel architecture. '
       'YOLOv8n led mAP50-95 at 0.529598 against 0.519570. YOLO26n reduced measured NUC four-thread p95 '
       'from 164.389 to 88.471 ms, about 46%, while missing full noninferiority. '
       'The production and historical 320 models are latency anchors only; their accuracy protocols are incompatible. '
       'No new detector replaced production.', '10-11'),
      ('Private field semantics make sharing useful','multitask-sharing',
       'Expanded trusted supervision repaired much of the early shallow-adapter collapse. '
       'A private field-context hierarchy then produced near-specialist selected-seed field quality while preserving '
       'the efficient frozen detector. Matched controls did not establish that shared-prefix adaptation was necessary. '
       'Final AC-verified laptop timing gave a pooled 6.55% reduction, not a consistent 10% gain. '
       'Seed variability and strict raw-parity failures remain. All these quality comparisons use open development data.', '12-13'),
      ('Simulation enables causal downstream tests',None,
       'The deterministic ROS Noetic / Gazebo 11 infrastructure contains a 16-scenario, 2,250-frame perception benchmark '
       'with synchronized camera geometry and supported synthetic annotations. Robot/X box scores are unsupported. '
       'A separate 4,230-frame stress study showed that soft field-context ball reranking could remove 300 off-field accepts '
       'and improve recall. On untouched replay it reduced recall by 1.708 percentage points and boundary recall by 9.910 points. '
       'The constructive result is a reusable benchmark and a better contextual formulation, without unsupported integration.', '14-15'),
      ('Separate measurement utility from map identity','localization-causal',
       'A neutral observation gate let incompatible particles outrank compatible negative log likelihoods. '
       'After the minimal repair, exact typed L/T observations reduced median simulated translation error from 0.971 to 0.299 m. '
       'Actual YOLOv8 measurements with oracle identity reached 0.437 m, recovering 79.5% of median oracle gain. '
       'Learned association had a 4.372 m p95. Correct identity is a diagnostic intervention, not a deployable sensor. '
       'The C++ correction passed historical component, Gazebo and real-Chape runtime tests. It is not claimed merged.', '16-18'),
      ('Improve tracking, then confront tail errors','tracking-and-lines',
       'A sensor-aided robust observation mixture reduced typical translation from 0.629 to 0.111 m, while p95 rose '
       'from 2.066 to 2.911 m. Sensors and odometry were explicitly synthetic. On a separate reused development suite, '
       'measured lines reduced median/p95 translation from 0.136/0.486 to 0.034/0.243 m. '
       'These protocols cannot be pooled. Later audits identify map/plane consistency and 180-degree symmetry as important '
       'limitations, alongside local failures even without a complete heading loss.', '19-21'),
      ('Engineering handoff and remaining decisions',None,
       'The repository separates runnable public smoke examples, model inference, and scientific reproduction requiring '
       'separately licensed data. Models are identified by hashes and contracts. Strict parity warnings remain explicit. '
       'The isolated C++ correction is the reviewable software contribution. Learned association and field-context filtering '
       'remain research modules. Independent real-robot accuracy and pose ground truth, a team timing budget, and robust '
       'symmetry-aware recovery are the next integration dependencies. No protected data or new experiment is part of publication.', '6-21')]
    story=[]
    for i,(title,figure,text,refs) in enumerate(sections):
        if i and i%2==0:story.append(PageBreak())
        elif i:story.append(Spacer(1,22))
        story += [Paragraph(title,ST['OverviewTitle']),Paragraph(text,ST['Text'])]
        if figure:
            from PIL import Image as PILImage
            image=ROOT/'results/figures'/f'{figure}.png';w,h=PILImage.open(image).size
            story += [Spacer(1,8),Image(str(image),width=491,height=491*h/w)]
        story += [Spacer(1,8),Paragraph('Sources: technical reports ['+refs+']. Figures: original aggregate redraws; see figure-provenance.json. '
            'All source tables and report editions are available in the companion repository.',ST['Small'])]
    build(a.output/'engineering-overview.pdf',story,'RoboCup Humanoid Perception: Engineering Overview')
    print(f'Rendered {len(mapping)} reports and a compact engineering overview.')


if __name__=='__main__':main()
