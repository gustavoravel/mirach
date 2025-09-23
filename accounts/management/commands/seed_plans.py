from django.core.management.base import BaseCommand
from accounts.models import Plan


class Command(BaseCommand):
    help = 'Seed default plans (Free, Pro, Business)'

    def handle(self, *args, **options):
        Plan.objects.update_or_create(code='free', defaults={
            'name': 'Free',
            'is_enterprise': False,
            'max_projects': 1,
            'max_datasets': 2,
            'max_rows_per_dataset': 50000,
            'monthly_predictions': 5,
            'priority': 'low',
            'includes_advanced_models': False,
            'includes_backtesting': False,
            'includes_exports': False,
        })
        Plan.objects.update_or_create(code='pro', defaults={
            'name': 'Pro',
            'is_enterprise': False,
            'max_projects': 9999,
            'max_datasets': 9999,
            'max_rows_per_dataset': 5000000,
            'monthly_predictions': 500,
            'priority': 'high',
            'includes_advanced_models': True,
            'includes_backtesting': True,
            'includes_exports': True,
        })
        Plan.objects.update_or_create(code='business', defaults={
            'name': 'Business',
            'is_enterprise': True,
            'max_projects': 999999,
            'max_datasets': 999999,
            'max_rows_per_dataset': 10000000,
            'monthly_predictions': 10000,
            'priority': 'high',
            'includes_advanced_models': True,
            'includes_backtesting': True,
            'includes_exports': True,
        })
        self.stdout.write(self.style.SUCCESS('Plans seeded'))


