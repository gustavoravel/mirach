from django.core.management.base import BaseCommand
from predictions.services import initialize_prediction_models


class Command(BaseCommand):
    help = 'Initialize prediction models in the database'

    def handle(self, *args, **options):
        self.stdout.write('Initializing prediction models...')
        
        try:
            initialize_prediction_models()
            self.stdout.write(
                self.style.SUCCESS('Successfully initialized prediction models!')
            )
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error initializing models: {str(e)}')
            )
