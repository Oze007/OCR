from django.http import JsonResponse
import cv2
from ultralytics import YOLO
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import os
import statistics
from django.views.decorators.csrf import csrf_exempt
from detectron2 import model_zoo
from pymongo import MongoClient
from django.conf import settings
from datetime import datetime
import json
import os

# Connexion à MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["ocr_db"]
collection = db["Matricules"]

# Charger les codes des préfectures depuis un fichier JSON
def load_prefectures():
    # Chemin vers le fichier JSON
    current_directory = os.path.dirname(_file_)
    file_path = os.path.join(current_directory, 'city_codes.json')  # Met à jour le chemin

    # Charger le contenu du fichier
    with open(file_path, 'r', encoding='utf-8') as file:
        prefectures = json.load(file)


    return prefectures

# Path to the YOLO model
current_directory = os.path.dirname(_file_)
license_plate_detector_path = os.path.join(current_directory, 'license_plate_detector.pt')

# Load the YOLO model
license_plate_detector = YOLO(license_plate_detector_path)

class PlateOCR:
    def _init_(self):
        self._cfg = get_cfg()
        self._predictor = self._makePredictor()
        self._characters = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "h", "w", "d", "p", "waw", "j", "t"]
    
    def _makePredictor(self):
        self._cfg.MODEL.DEVICE = "cpu"
        self._cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        self._cfg.SOLVER.IMS_PER_BATCH = 2
        self._cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  
        self._cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20
        self._cfg.MODEL.WEIGHTS = os.path.join(current_directory, 'weights', 'plate_ocr', 'model_final.pth')
        self._cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
        return DefaultPredictor(self._cfg)

    def predict(self, image):
        return self._predictor(image)
    
    def characterBoxes(self, output):
        boxes = output['instances'].pred_boxes.tensor.cpu().numpy().tolist() 
        scores = output['instances'].scores.numpy().tolist()
        classes = output['instances'].pred_classes.to('cpu').tolist()
        characters = {i: {"character": self._characters[classes[i]], "score": scores[i], "boxes": boxes[i]} for i in range(len(scores))}
        return characters

    def postProcess(self, image, output):
        if not output:
            return {'plate': image[:-4], 'plate_string': ''}
        y_mins = [value['boxes'][1] for value in output.values()]
        median_y_mins = statistics.median(y_mins)
        top_characters = {key: value for key, value in output.items() if value['boxes'][3] <= median_y_mins}
        bottom_characters = {key: value for key, value in output.items() if value['boxes'][3] > median_y_mins}
        
        sorted_top_characters = sorted(top_characters.items(), key=lambda e: e[1]['boxes'][0])
        sorted_bottom_characters = sorted(bottom_characters.items(), key=lambda e: e[1]['boxes'][0])
        
        plate_ocr = "".join([item[1]['character'] for item in sorted_bottom_characters + sorted_top_characters])
        return {'plate': image[:-4], 'plate_string': plate_ocr}

@csrf_exempt
def process_license_plate(request):
    image_path = request.GET.get('path', None)
    
    if image_path is None:
        return JsonResponse({'error': 'Image path not provided in URL parameters.'}, status=400)
    
    try:
        frame = cv2.imread(image_path)
        license_plates = license_plate_detector(frame)[0]
        plate_ocr = PlateOCR()
        
        plate_results = []
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2)]
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            output = plate_ocr.predict(license_plate_crop)
            characters = plate_ocr.characterBoxes(output)
            plate_ocr_string = plate_ocr.postProcess(image_path, characters)
            
            # Extraction des parties du matricule
            plate_string = plate_ocr_string['plate_string']
            
            # Trouver la position de la première lettre dans le matricule
            letter_index = next((i for i, c in enumerate(plate_string) if c.isalpha()), None)
            
            if letter_index is not None:
                # Les premiers chiffres (1 à 5 chiffres avant la lettre)
                first_numbers = plate_string[:letter_index]
                first_numbers = first_numbers[:5]

                # Les lettres (jusqu'à 3 lettres consécutives après les chiffres)
                letters = ''
                i = letter_index
                while i < len(plate_string) and plate_string[i].isalpha() and len(letters) < 3:
                    letters += plate_string[i]
                    i += 1

                # Limiter à 2 chiffres après la lettre
                last_numbers = plate_string[letter_index + len(letters):letter_index + len(letters) + 2]
                last_numbers = last_numbers[:2]
                
                # Charger les codes des préfectures
                prefectures = load_prefectures()

                # Comparer last_numbers avec les codes des préfectures
                city = prefectures.get(last_numbers, "Inconnu")
                print(f"LastNumbers: {last_numbers} correspond à la ville: {city}")
                
                # Stocker les résultats dans MongoDB
                plate_data = {
                    #"image_path": image_path,
                    "matricule": plate_string,
                    "timestamp": datetime.now().isoformat(),
                    "first_numbers": first_numbers,
                    "letters": letters,
                    "last_numbers": last_numbers,
                    "city": city
                }
                db.plates.insert_one(plate_data)

                # Collecter les résultats pour les renvoyer dans la réponse
                plate_results.append({
                    'Matricule': plate_string,
                    'FirstNumbers': first_numbers,
                    'Letters': letters,
                    'LastNumbers': last_numbers,
                    'City': city
                })
        
        return JsonResponse({'plates': plate_results}, safe=False)
    
    except Exception as e:
        return JsonResponse({'error': f'An error occurred: {str(e)}'}, status=500)