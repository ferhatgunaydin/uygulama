from flask import Flask, request, render_template, Response, send_file
import os
import numpy as np
import cv2
import csv
from download_model import download_model_if_not_exists
download_model_if_not_exists()
from tensorflow.keras.models import load_model
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, Frame
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, Frame
from reportlab.lib.styles import getSampleStyleSheet
from flask import send_file


# Flask ayarları
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/results'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)

# Veritabanı modeli
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(50))
    image_filename = db.Column(db.String(100))
    prediction = db.Column(db.String(10))
    segment_filename = db.Column(db.String(100), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Modelleri yükle
clf_model = load_model("models/classifier_model.keras")
seg_model = load_model("models/segmenter_model.h5")

# Ana sayfa
@app.route("/")
def index():
    recent = Prediction.query.order_by(Prediction.timestamp.desc()).limit(5).all()
    return render_template("index.html", recent=recent)

# Kayıtlar sayfası + filtreleme
@app.route("/records", methods=["GET", "POST"])
def records():
    query = Prediction.query.order_by(Prediction.timestamp.desc())
    if request.method == "POST":
        pid = request.form.get("patient_id")
        date = request.form.get("date")
        if pid:
            query = query.filter(Prediction.patient_id.like(f"%{pid}%"))
        if date:
            query = query.filter(Prediction.timestamp.like(f"{date}%"))
    all_records = query.all()
    return render_template("records.html", records=all_records)

# CSV dışa aktarım
@app.route("/export")
def export_csv():
    records = Prediction.query.order_by(Prediction.timestamp.desc()).all()

    def generate():
        yield "Hasta ID,Görsel,Tahmin,Segmentasyon,Tarih\n"
        for r in records:
            seg = r.segment_filename if r.segment_filename else "-"
            yield f"{r.patient_id},{r.image_filename},{r.prediction},{seg},{r.timestamp}\n"

    return Response(generate(), mimetype='text/csv',
                    headers={"Content-Disposition": "attachment;filename=kayitlar.csv"})

# PDF raporu oluşturma
@app.route("/pdf/<int:record_id>")
def generate_pdf(record_id):
    record = Prediction.query.get_or_404(record_id)
    pdf_path = f"static/results/report_{record.id}.pdf"

    # Font kaydı (ilk çalıştırmada)
    pdfmetrics.registerFont(TTFont('TurkceFont', 'fonts/DejaVuSans.ttf'))

    from reportlab.pdfgen import canvas
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    styles = getSampleStyleSheet()
    styles["Normal"].fontName = 'TurkceFont'

    # Başlık
    c.setFont("TurkceFont", 20)
    c.drawCentredString(width / 2, height - 50, "Beyin BT İnme Analizi Raporu")

    # Tarih
    c.setFont("TurkceFont", 10)
    c.drawRightString(width - 40, height - 70, f"Tarih: {record.timestamp.strftime('%Y-%m-%d %H:%M')}")

    # Hasta bilgileri kutusu
    c.setFillColor(colors.lightgrey)
    c.rect(30, height - 150, width - 60, 70, fill=True, stroke=False)
    c.setFillColor(colors.black)
    c.setFont("TurkceFont", 12)
    c.drawString(40, height - 100, f"Hasta ID: {record.patient_id}")
    c.drawString(40, height - 120, f"Tahmin Sonucu: {record.prediction}")

    # Segmentasyon görseli (varsa)
    if record.segment_filename:
        segment_img_path = os.path.join("static/results", record.segment_filename)
        try:
            c.drawImage(segment_img_path, 40, height - 380, width=300, height=200)
            c.setFont("TurkceFont", 10)
            c.drawString(40, height - 390, "🔍 Segmentasyon Görseli")
        except:
            c.setFont("TurkceFont", 10)
            c.drawString(40, height - 390, "(Segmentasyon görseli yüklenemedi)")

    # Açıklama
    explanation = Paragraph(
        "Bu rapor, yüklenen beyin BT görüntüsünde inme tespiti amacıyla geliştirilen yapay zeka modelinin çıktısını içermektedir. Segmentasyon, tespit edilen anormal alanları kırmızı olarak işaretlemektedir.",
        styles["Normal"]
    )
    frame = Frame(40, 120, width - 80, 100, showBoundary=0)
    frame.addFromList([explanation], c)

    # İmza çizgisi
    c.line(40, 80, 200, 80)
    c.setFont("TurkceFont", 10)
    c.drawString(40, 65, "Doktor / Radyolog İmzası")

    c.save()
    return send_file(pdf_path, as_attachment=True)

# Tahmin işlemi
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    patient_id = request.form["patient_id"]
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    img = cv2.imread(filepath)
    img_resized = cv2.resize(img, (224, 224)) / 255.0
    input_img = np.expand_dims(img_resized, axis=0)
    pred = clf_model.predict(input_img)[0][0]
    label = "İnme Var" if pred > 0.5 else "İnme Yok"

    segment_path = ""
    segment_filename = None
    if label == "İnme Var":
        seg_input = cv2.resize(img, (256, 256)) / 255.0
        seg_input = np.expand_dims(seg_input, axis=0)
        mask = seg_model.predict(seg_input)[0]
        mask = (mask > 0.5).astype(np.uint8)[..., 0] * 255
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

        colored_mask = np.zeros_like(img)
        colored_mask[mask > 0] = [0, 0, 255]
        overlay = cv2.addWeighted(img, 0.8, colored_mask, 0.2, 0)

        segment_filename = f"seg_{filename}"
        segment_path = os.path.join(app.config['UPLOAD_FOLDER'], segment_filename)
        cv2.imwrite(segment_path, overlay)

    record = Prediction(
        patient_id=patient_id,
        image_filename=filename,
        prediction=label,
        segment_filename=segment_filename
    )
    db.session.add(record)
    db.session.commit()

    result_html = f"""
        <h2>Hasta ID: {patient_id}</h2>
        <h3>Tahmin: {label}</h3>
        <h4>Orijinal Görsel</h4>
        <img src="/static/results/{filename}" width="400"><br>
    """
    if segment_path:
        result_html += f"<h4>Segmentasyon</h4><img src='/static/results/{segment_filename}' width='400'>"
    result_html += f"<br><a href='/pdf/{record.id}'>📄 PDF Raporu İndir</a>"
    result_html += "<br><a href='/'>⬅ Ana Sayfa</a> | <a href='/records'>📖 Kayıtlar</a>"
    return result_html

# Uygulamayı başlat
if __name__ == "__main__":
    os.makedirs("static/results", exist_ok=True)
    with app.app_context():
        db.create_all()
    app.run(debug=True)
