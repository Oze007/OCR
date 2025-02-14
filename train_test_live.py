import cv2
import easyocr
from ultralytics import YOLO

# Charger le détecteur de texte
reader = easyocr.Reader(['fr'])

# Charger le modèle YOLOv8 pré-entraîné pour la détection des véhicules


model_vehicle = YOLO("yolov8n.pt")
model_license_plate = 'best1.pt'


# Charger le modèle YOLOv8 pour la détection des plaques d'immatriculation

# Charger la vidéo
video_path = '../PFE/videos/test1111.mp4'
cap = cv2.VideoCapture(video_path)

matricule_captures = {}

# Variables pour compter les captures
capture_count = 0


# ...

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Détecter les véhicules avec YOLOv8
    results_vehicle = model_vehicle(frame)

    # Exécuter la détection des plaques d'immatriculation pour toutes les prédictions
    results_license_plate = model_license_plate(frame)

    # Vérifier si une plaque d'immatriculation est détectée avec le modèle YOLOv8 personnalisé
    for result_license_plate in results_license_plate:
        for pred_license_plate in result_license_plate.boxes:
            # Récupérer les coordonnées de la plaque d'immatriculation
            x1, y1, x2, y2 = map(int, pred_license_plate.xyxy[0])
            # Extraire la région de la plaque d'immatriculation de l'image
            license_plate_region = frame[y1:y2, x1:x2]

            # Récupérer le matricule reconnu avec EasyOCR
            results_ocr = reader.readtext(license_plate_region)
            if results_ocr:
                text_ocr = results_ocr[0][-1]

                # Vérifier si le matricule est déjà dans le dictionnaire
                if text_ocr not in matricule_captures:
                    matricule_captures[text_ocr] = 0

                # Vérifier si le nombre de captures pour ce matricule est inférieur à 2
                if matricule_captures[text_ocr] < 2:
                    # Faire une capture d'écran du matricule
                    capture_count += 1

                    capture_path = f'../PFE/captures/capture_{capture_count}.png'
                    cv2.imwrite(capture_path, license_plate_region)
                    print(f'Capture enregistrée pour le matricule {text_ocr}: {capture_path}')

                    # Utiliser EasyOCR pour reconnaître les caractères de la plaque d'immatriculation
                    print(f'Matricule lu : {text_ocr}')

                    # Mettre à jour le nombre de captures pour ce matricule
                    matricule_captures[text_ocr] += 1

    # Afficher la vidéo avec les résultats
    cv2.imshow('Vehicle and License Plate Detection', frame)

    # Attendre 1 milliseconde et vérifier si l'utilisateur appuie sur 'q' pour quitter
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# ...

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()