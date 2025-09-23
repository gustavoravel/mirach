from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from projects.models import Project
from datasets.models import Dataset, ColumnMapping
from predictions.models import PredictionModel
from django.core.files.base import ContentFile
import pandas as pd
import io


class Command(BaseCommand):
    help = 'Create sample project, dataset, and models for onboarding'

    def handle(self, *args, **options):
        user, _ = User.objects.get_or_create(username='demo', defaults={'email': 'demo@example.com'})
        user.set_password('demo1234')
        user.save()

        project, _ = Project.objects.get_or_create(name='Projeto Exemplo', owner=user)

        # Create simple time series dataset
        dates = pd.date_range(start='2024-01-01', periods=60, freq='D')
        values = [i + 10 for i in range(60)]
        df = pd.DataFrame({'date': dates, 'sales': values})
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
        buffer.seek(0)

        dataset, _ = Dataset.objects.get_or_create(
            name='Vendas Exemplo.xlsx',
            project=project,
            uploaded_by=user,
        )
        dataset.file.save('vendas_exemplo.xlsx', ContentFile(buffer.read()))
        dataset.total_rows = len(df)
        dataset.total_columns = len(df.columns)
        dataset.column_names = list(df.columns)
        dataset.status = 'processed'
        dataset.save()

        ColumnMapping.objects.get_or_create(dataset=dataset, column_name='date', defaults={'column_type': 'timestamp', 'data_type': 'datetime'})
        ColumnMapping.objects.get_or_create(dataset=dataset, column_name='sales', defaults={'column_type': 'target', 'data_type': 'float'})

        # Ensure some prediction models exist
        PredictionModel.objects.get_or_create(name='ARIMA', algorithm_type='arima', defaults={'description': 'ARIMA', 'parameters': {'auto_order': True}})
        PredictionModel.objects.get_or_create(name='ETS (Exponential Smoothing)', algorithm_type='ets', defaults={'description': 'ETS', 'parameters': {'seasonal': 'add', 'seasonal_periods': 7}})
        PredictionModel.objects.get_or_create(name='Prophet', algorithm_type='prophet', defaults={'description': 'Prophet', 'parameters': {}})

        self.stdout.write(self.style.SUCCESS('Onboarding seed created: user=demo / demo1234'))


