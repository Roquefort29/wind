import os
import logging
from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
from .lstm_model import evaluate_model

logger = logging.getLogger(__name__)


def index(request):
    return render(request, 'turbo/index.html')


def contacts(request):
    return render(request, 'turbo/contacts.html')


def predict(request):
    if request.method == 'POST' and request.FILES.get('data_file'):
        try:
            file = request.FILES['data_file']

            # Ensure the 'uploads' directory exists
            upload_dir = 'uploads'
            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)

            file_path = os.path.join(upload_dir, file.name)
            with open(file_path, 'wb+') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)

            mae, actual, predictions = evaluate_model(file_path)
            os.remove(file_path)  # Remove the file after processing

            return JsonResponse({
                'mae': mae,
                'actual': actual.tolist(),
                'predictions': predictions.tolist()
            })
        except Exception as e:
            logger.error(f"Error occurred during prediction: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)

    return render(request, 'turbo/predict.html')
