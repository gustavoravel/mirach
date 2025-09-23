from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import timedelta
from predictions.models import Prediction, PredictionResult


class Command(BaseCommand):
    help = 'Purge old predictions and results to reduce storage usage'

    def add_arguments(self, parser):
        parser.add_argument('--days', type=int, default=90, help='Age in days to purge (default: 90)')

    def handle(self, *args, **options):
        cutoff = timezone.now() - timedelta(days=options['days'])
        qs = Prediction.objects.filter(created_at__lt=cutoff)
        count = qs.count()
        # Delete results first
        PredictionResult.objects.filter(prediction__in=qs).delete()
        qs.delete()
        self.stdout.write(self.style.SUCCESS(f'Purged {count} predictions older than {options["days"]} days'))


