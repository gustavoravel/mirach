from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('predictions', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='prediction',
            name='status',
            field=models.CharField(choices=[('pending', 'Pendente'), ('queued', 'Na fila'), ('training', 'Treinando'), ('completed', 'Concluído'), ('failed', 'Falhou')], default='pending', max_length=20, verbose_name='Status'),
        ),
        migrations.AddField(
            model_name='prediction',
            name='progress',
            field=models.IntegerField(default=0, verbose_name='Progresso (%)'),
        ),
        migrations.AddField(
            model_name='prediction',
            name='estimated_completion',
            field=models.DateTimeField(blank=True, null=True, verbose_name='Conclusão Estimada'),
        ),
    ]


