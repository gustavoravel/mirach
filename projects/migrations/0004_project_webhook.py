from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('projects', '0003_invites_auditlog'),
    ]

    operations = [
        migrations.AddField(
            model_name='project',
            name='webhook_url',
            field=models.URLField(blank=True, verbose_name='Webhook de Conclusão'),
        ),
    ]


