import os
import json
import random
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Table,
    ListFlowable,
    Spacer,
    Frame,
    PageTemplate
)
from reportlab.lib.units import inch

# Configuration
NUM_SYNTHETIC_FILES = 50  # Number of PDFs to generate
OUTPUT_PDF_DIR = r"C:\Users\DIVYA\Desktop\JayShreeRam\Adobe-round-1a-Div\input"
OUTPUT_JSON_DIR = r"C:\Users\DIVYA\Desktop\JayShreeRam\Adobe-round-1a-Div\ground_truth"
NOISE_PROBABILITY = 0.2  # 20% chance of adding noise/edge cases
INVITATION_PROBABILITY = 0.2 # 20% chance of generating an invitation-style document

# Sample data for variability
LOREM_IPSUM = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
JAPANESE_TEXT = "セクション1: はじめに"
GERMAN_TEXT = "Abschnitt 1: Einführung"

def generate_random_style(level_name, base_size, is_invitation=False):
    """Generate randomized paragraph styles for headings and body"""
    font_choices = ["Helvetica-Bold", "Times-Bold", "Courier-Bold"]
    
    if is_invitation:
        # Invitation styles are often larger, more varied, and can be non-bold
        if level_name == "H1":
            font_size = random.randint(30, 60) # Very large
            text_color = random.choice([colors.black, colors.red, colors.blue, colors.green])
            font_name = random.choice(["Helvetica-Bold", "Times-Bold", "Courier-Bold", "Helvetica"]) # Can be non-bold
            alignment = random.choice([0, 1]) # Left or Center
        elif level_name == "H2": # For sub-prominent text like "YOU'RE INVITED"
            font_size = random.randint(20, 30)
            text_color = random.choice([colors.black, colors.darkblue])
            font_name = random.choice(["Helvetica", "Times-Bold"])
            alignment = random.choice([0, 1])
        else: # Body text in invitation
            font_size = random.randint(12, 18)
            text_color = colors.black
            font_name = random.choice(["Helvetica", "Times"])
            alignment = random.choice([0, 1])
        space_after = random.randint(15, 30) # More spacing
    else: # Standard document styles
        if level_name == "H1":
            font_size = random.randint(base_size - 3, base_size + 3)
            text_color = random.choice([colors.black, colors.darkblue, colors.darkgreen])
            font_name = random.choice(font_choices)
            alignment = random.choice([0, 1, 2])
        elif level_name == "H2":
            font_size = random.randint(base_size - 2, base_size + 2)
            text_color = random.choice([colors.black, colors.darkblue])
            font_name = random.choice(font_choices)
            alignment = random.choice([0, 1, 2])
        elif level_name == "H3":
            font_size = random.randint(base_size - 1, base_size + 1)
            text_color = colors.black
            font_name = random.choice(font_choices)
            alignment = random.choice([0, 1, 2])
        elif level_name == "H4":
            font_size = random.randint(base_size - 1, base_size + 1)
            text_color = colors.black
            font_name = random.choice(font_choices)
            alignment = random.choice([0, 1, 2])
        else: # Body text
            font_size = base_size
            text_color = colors.black
            font_name = "Helvetica"
            alignment = 0 # Left aligned for body
        space_after = random.randint(10, 20)

    return ParagraphStyle(
        name=f"Style_{level_name}_{random.randint(0,1000)}", # Unique name for each style
        fontSize=font_size,
        textColor=text_color,
        fontName=font_name,
        alignment=alignment,
        spaceAfter=space_after
    )

def add_noise(text):
    """Add random noise to text (OCR simulation)"""
    if random.random() < NOISE_PROBABILITY:
        # Introduce more varied noise
        text = text.replace("o", "0").replace("e", "3").replace("a", "@").replace("l", "1").replace("s", "5")
        if random.random() < 0.1: # Small chance of adding extra space
            text = text.replace(" ", "  ")
    return text

def generate_invitation_pdf(filename):
    """Generate an invitation-style PDF"""
    doc = SimpleDocTemplate(filename, pagesize=letter)
    story = []
    outline = []
    
    # Use a fixed frame to center content for invitation
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height,
                  leftPadding=0, bottomPadding=0, rightPadding=0, topPadding=0,
                  showBoundary=0) # No boundary for final PDF
    
    # Main Title (e.g., TOPJUMP)
    main_title_text = random.choice(["TOPJUMP", "PARTY ZONE", "FUNLAND"])
    main_title_style = generate_random_style("H1", 60, is_invitation=True)
    story.append(Paragraph(main_title_text, main_title_style))
    
    # Sub-title (e.g., TRAMPOLINE PARK)
    sub_title_text = random.choice(["TRAMPOLINE PARK", "ADVENTURE ARENA", "BIRTHDAY BASH"])
    sub_title_style = generate_random_style("H2", 30, is_invitation=True)
    story.append(Paragraph(sub_title_text, sub_title_style))
    story.append(Spacer(1, 0.5 * inch))

    # Invitation phrase (e.g., YOU'RE INVITED TO A PARTY)
    invite_phrase_text = "YOU'RE INVITED TO A"
    invite_phrase_style = generate_random_style("H1", 40, is_invitation=True) # Large, prominent
    story.append(Paragraph(invite_phrase_text, invite_phrase_style))
    
    party_word_text = "PARTY"
    party_word_style = generate_random_style("H1", 70, is_invitation=True) # Even larger
    story.append(Paragraph(party_word_text, party_word_style))
    story.append(Spacer(1, 0.5 * inch))

    # Key details (FOR:, DATE:, TIME:, ADDRESS:) - often bold, smaller than main headings
    details = [
        ("FOR:", "John Doe's Birthday"),
        ("DATE:", "July 28, 2025"),
        ("TIME:", "3:00 PM - 5:00 PM"),
        ("ADDRESS:", "123 Fun Street, Partyville, USA")
    ]
    for label, value in details:
        story.append(Paragraph(f"<b>{label}</b> {value}", generate_random_style("BodyText", 14, is_invitation=True)))
        story.append(Spacer(1, 0.1 * inch))
    story.append(Spacer(1, 0.3 * inch))

    # Important notes (often smaller, but bold or distinct)
    important_notes = [
        "CLOSED TOED SHOES ARE REQUIRED",
        "PLEASE VISIT WEBSITE TO FILL OUT WAIVER"
    ]
    for note in important_notes:
        story.append(Paragraph(f"<b>{note}</b>", generate_random_style("BodyText", 10, is_invitation=True)))
        story.append(Spacer(1, 0.05 * inch))
    story.append(Spacer(1, 0.2 * inch))

    # Call to action (e.g., HOPE TO SEE YOU THERE!) - this is the ground truth H1
    call_to_action_text = "HOPE TO SEE YOU THERE!"
    call_to_action_style = generate_random_style("H1", 24, is_invitation=True) # Prominent, but not largest
    story.append(Paragraph(call_to_action_text, call_to_action_style))
    outline.append({"level": "H1", "text": call_to_action_text, "page": 0}) # THIS IS OUR GT H1

    # Website
    website_text = "WWW.PARTYPLACE.COM"
    website_style = generate_random_style("BodyText", 12, is_invitation=True)
    story.append(Paragraph(website_text, website_style))

    doc.build(story, canvasmaker=lambda canvas, doc: canvas) # No page numbers for invitation

    # The actual ground truth for this type of document is very specific
    # In the original TOPJUMP PDF, "HOPE TO SEE YOU THERE!" is the only "heading"
    # that makes sense in a hierarchical context.
    # We will return the specific outline for this type of document.
    return outline


def generate_standard_pdf(filename):
    """Generate a standard academic/report-style PDF"""
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyTextCustom',
                              parent=styles['BodyText'],
                              fontSize=random.randint(10, 12),
                              leading=random.randint(12, 15),
                              spaceAfter=random.randint(5, 10)))

    story = []
    outline = []
    
    # Random title with 10% chance of being multilingual
    title_text = f"Document Title {random.randint(1, 100)}: A Comprehensive Study"
    if random.random() < 0.1:
        title_text = random.choice([JAPANESE_TEXT, GERMAN_TEXT]) + f" ({random.randint(1, 100)})"
    
    title_style = generate_random_style("H1", 24)
    story.append(Paragraph(title_text, title_style))
    outline.append({"level": "H1", "text": title_text, "page": 0})
    
    for page_num in range(random.randint(3, 15)):
        for _ in range(random.randint(2, 5)): # 2-5 headings per page
            level_choice = random.choice(["H1", "H2", "H3", "H4"]) 
            
            base_sizes = {"H1": 24, "H2": 18, "H3": 14, "H4": 12}
            heading_style = generate_random_style(level_choice, base_sizes[level_choice])

            prefix = ""
            if level_choice == "H1": prefix = f"{random.randint(1, 10)}."
            elif level_choice == "H2": prefix = f"{random.randint(1, 10)}.{random.randint(1, 5)}"
            elif level_choice == "H3": prefix = f"{random.randint(1, 10)}.{random.randint(1, 5)}.{random.randint(1, 3)}"
            elif level_choice == "H4": prefix = f"{random.randint(1, 10)}.{random.randint(1, 5)}.{random.randint(1, 3)}.{random.randint(1, 2)}"
            
            heading_topic = random.choice([
                'Introduction', 'Methodology', 'Results', 
                'Discussion', 'Conclusion', 'References',
                'Analysis', 'Findings', 'Implementation', 'Future Work'
            ])
            
            heading_text = f"{prefix} {heading_topic}"
            
            heading_text = add_noise(heading_text)
            if random.random() < 0.3: heading_text = heading_text.upper()
            if random.random() < 0.1: heading_text += ":"
            
            story.append(Paragraph(heading_text, heading_style))
            outline.append({"level": level_choice, "text": heading_text, "page": page_num})
            
            if random.random() < 0.7:
                story.append(Paragraph(LOREM_IPSUM * random.randint(1, 3), styles["BodyTextCustom"]))
                story.append(Spacer(1, random.uniform(0.1, 0.3) * inch))
            
            if random.random() < 0.3:
                items = [f"{random.choice(['•', '-', '->'])} Item {i} {LOREM_IPSUM.split('.')[0]}" for i in range(1, random.randint(2, 5))]
                story.append(ListFlowable([Paragraph(item, styles["BodyTextCustom"]) for item in items]))
                story.append(Spacer(1, random.uniform(0.1, 0.2) * inch))
            
            if random.random() < 0.2:
                data = [[f"Header {i+1}" for i in range(3)], *[[str(random.randint(1, 100)) for _ in range(3)] for _ in range(3)]]
                story.append(Table(data, style=[("GRID", (0,0), (-1,-1), 1, colors.grey), ("BACKGROUND", (0,0), (-1,0), colors.lightgrey)]))
                if random.random() < 0.5:
                    caption_text = f"Table {page_num+1}.{random.randint(1, 5)}: Sample data table on {random.choice(['performance', 'trends', 'metrics'])}"
                    story.append(Paragraph(caption_text, ParagraphStyle(name="Caption", fontSize=random.randint(9,11), textColor=colors.grey, spaceAfter=5)))
                story.append(Spacer(1, random.uniform(0.1, 0.2) * inch))
    
    doc.build(story)
    return outline

def save_ground_truth(outline, pdf_filename):
    """Save ground truth JSON with metadata"""
    json_filename = os.path.basename(pdf_filename).replace(".pdf", ".json")
    json_path = os.path.join(OUTPUT_JSON_DIR, json_filename)
    
    title_from_outline = outline[0]["text"] if outline else ""

    metadata = {
        "title": title_from_outline,
        "outline": outline,
        "metadata": {
            "generator": "synthetic_data.py",
            "variability_features": [
                "mixed_headings",
                "random_body_text",
                "tables",
                "lists",
                "noise_injection",
                "multilingual_headings",
                "variable_spacing",
                "H4_levels_included"
            ]
        }
    }
    
    with open(json_path, "w", encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

def validate_directories():
    """Ensure output directories exist"""
    os.makedirs(OUTPUT_PDF_DIR, exist_ok=True)
    os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)
    print(f"PDFs will be saved to: {OUTPUT_PDF_DIR}")
    print(f"JSONs will be saved to: {OUTPUT_JSON_DIR}")

def main():
    validate_directories()
    
    for i in range(NUM_SYNTHETIC_FILES):
        pdf_filename = os.path.join(OUTPUT_PDF_DIR, f"synthetic_{i}.pdf")
        
        if random.random() < INVITATION_PROBABILITY:
            print(f"Generating invitation-style {pdf_filename}...")
            outline = generate_invitation_pdf(pdf_filename)
        else:
            print(f"Generating standard-style {pdf_filename}...")
            outline = generate_standard_pdf(pdf_filename)
        
        save_ground_truth(outline, pdf_filename)

if __name__ == "__main__":
    main()
    print(f"\nGenerated {NUM_SYNTHETIC_FILES} PDFs + JSONs with:")
    print("- Randomized headings (H1-H4) with varied styles")
    print("- Body text, lists, and tables")
    print("- Noise injection (typos, all-caps, extra spaces)")
    print("- Multilingual support (Japanese/German)")
    print("- Configurable variability via NOISE_PROBABILITY")
    print("- Increased heading density per page")
    print(f"- {INVITATION_PROBABILITY*100}% of documents are invitation-style")
