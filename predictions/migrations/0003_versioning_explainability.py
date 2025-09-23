from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('predictions', '0002_progress_eta_status'),
    ]

    operations = [
        migrations.AddField(
            model_name='prediction',
            name='dataset_snapshot',
            field=models.JSONField(blank=True, default=dict, verbose_name='Snapshot do Dataset'),
        ),
        migrations.AddField(
            model_name='prediction',
            name='dataset_version',
            field=models.CharField(blank=True, max_length=50, verbose_name='Versão do Dataset'),
        ),
        migrations.AddField(
            model_name='prediction',
            name='explainability',
            field=models.JSONField(blank=True, default=dict, verbose_name='Explainability'),
        ),
        migrations.AddField(
            model_name='prediction',
            name='model_version',
            field=models.CharField(blank=True, max_length=50, verbose_name='Versão do Modelo'),
        ),
    ]


