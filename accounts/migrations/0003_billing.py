from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0002_api_tokens'),
        ('auth', '0012_alter_user_first_name_max_length'),
    ]

    operations = [
        migrations.CreateModel(
            name='Plan',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('code', models.CharField(max_length=30, unique=True)),
                ('name', models.CharField(max_length=100)),
                ('is_enterprise', models.BooleanField(default=False)),
                ('max_projects', models.IntegerField(default=1)),
                ('max_datasets', models.IntegerField(default=2)),
                ('max_rows_per_dataset', models.IntegerField(default=50000)),
                ('monthly_predictions', models.IntegerField(default=5)),
                ('priority', models.CharField(default='low', max_length=10)),
                ('includes_advanced_models', models.BooleanField(default=False)),
                ('includes_backtesting', models.BooleanField(default=False)),
                ('includes_exports', models.BooleanField(default=False)),
            ],
        ),
        migrations.CreateModel(
            name='Subscription',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('started_at', models.DateTimeField(default=django.utils.timezone.now)),
                ('trial_ends_at', models.DateTimeField(blank=True, null=True)),
                ('is_active', models.BooleanField(default=True)),
                ('plan', models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, to='accounts.plan')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='subscriptions', to='auth.user')),
            ],
        ),
        migrations.CreateModel(
            name='UsageEvent',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('event_type', models.CharField(max_length=50)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('metadata', models.JSONField(blank=True, default=dict)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='usage_events', to='auth.user')),
            ],
        ),
        migrations.CreateModel(
            name='CreditBalance',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('balance', models.IntegerField(default=0)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='credit_balances', to='auth.user')),
            ],
        ),
    ]


