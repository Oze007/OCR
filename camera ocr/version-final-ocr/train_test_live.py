import cv2
from ultralytics import YOLO
import os
import time
import easyocr

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])  # 'en' for English, add other languages if needed

def detect_vehicles_and_capture_plates(vehicle_model_path, plate_model_path, output_folder):
    # Charger les modèles YOLO
    model_vehicle = YOLO(vehicle_model_path)
    model_license_plate = YOLO(plate_model_path)
    
    # Créer le dossier de captures si inexistant
    os.makedirs(output_folder, exist_ok=True)
    
    # Ouvrir la caméra (0 pour la caméra par défaut)
    cap = cv2.VideoCapture(0)
    
    # Vérifier si la caméra s'ouvre correctement
    if not cap.isOpened():
        print("Erreur : Impossible d'ouvrir la caméra")
        return
    
    capture_count = 0
    start_time = time.time()  # Temps de début
    
    # Affichage du flux vidéo
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur : Impossible de lire le flux vidéo")
            break
        
        # Détecter les véhicules
        results_vehicle = model_vehicle(frame)
        
        for result_vehicle in results_vehicle:
            for pred_vehicle in result_vehicle.boxes:
                x1, y1, x2, y2 = map(int, pred_vehicle.xyxy[0])
                vehicle_region = frame[y1:y2, x1:x2]
                
                # Détecter les plaques d'immatriculation dans la région du véhicule
                results_license_plate = model_license_plate(vehicle_region)
                
                for result in results_license_plate:
                    for pred_license_plate in result.boxes:
                        x1_lp, y1_lp, x2_lp, y2_lp = map(int, pred_license_plate.xyxy[0])
                        
                        # Ensure the bounding box is within the vehicle region
                        height, width, _ = vehicle_region.shape
                        x1_lp = max(0, x1_lp)
                        y1_lp = max(0, y1_lp)
                        x2_lp = min(width, x2_lp)
                        y2_lp = min(height, y2_lp)
                        
                        license_plate_region = vehicle_region[y1_lp:y2_lp, x1_lp:x2_lp]
                        
                        # Check if the license plate region is valid
                        if license_plate_region.size == 0:
                            print("Erreur : Région de la plaque d'immatriculation vide ou invalide")
                            continue
                        
                        # Utiliser EasyOCR pour lire la plaque d'immatriculation
                        result_text = reader.readtext(license_plate_region)
                        
                        if result_text:
                            plate_text = result_text[0][1]  # Premier texte trouvé
                            print(f"Plaque d'immatriculation détectée: {plate_text}")
                            
                            # Enregistrer la capture d'écran
                            capture_count += 1
                            timestamp = int(time.time())  # Use a timestamp for unique filenames
                            capture_path = os.path.join(output_folder, f'capture_{timestamp}_{capture_count}.png')
                            cv2.imwrite(capture_path, license_plate_region)
                            print(f'Capture enregistrée : {capture_path}')
                            
                            # Stop the loop after capturing one image
                            cap.release()
                            cv2.destroyAllWindows()
                            return
                        
                        # Attendre 1 seconde avant la prochaine capture
                        time.sleep(1)
        
        # Afficher la vidéo avec détection
        cv2.imshow('Détection des Véhicules et Plaques', frame)
        
        # Quitter si la touche 'q' est pressée
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Arrêter après 5 secondes
        if time.time() - start_time > 5:
            print("Temps écoulé : 5 secondes")
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Paramètres
vehicle_model_path = r"C:\Users\othma\OneDrive\Bureau\HACKATHON\version-final-ocr\yolov8n.pt"
plate_model_path =  r"C:\Users\othma\OneDrive\Bureau\HACKATHON\version-final-ocr\best200.pt"  
output_folder = r"C:\Users\othma\OneDrive\Bureau\HACKATHON\version-final-ocr\capture"

# Exécuter la détection en temps réel
detect_vehicles_and_capture_plates(vehicle_model_path, plate_model_path, output_folder)