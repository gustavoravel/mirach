from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('projects', '0004_project_webhook'),
    ]

    operations = [
        migrations.AddField(
            model_name='project',
            name='priority',
            field=models.IntegerField(default=0, verbose_name='Prioridade do Projeto'),
        ),
    ]


