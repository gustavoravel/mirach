# Generated manually for data_profile + ai_interpretation

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('datasets', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='dataset',
            name='data_profile',
            field=models.JSONField(blank=True, default=dict, verbose_name='Perfil dos Dados'),
        ),
        migrations.AddField(
            model_name='dataset',
            name='ai_interpretation',
            field=models.JSONField(blank=True, default=dict, verbose_name='Interpretação IA'),
        ),
    ]
