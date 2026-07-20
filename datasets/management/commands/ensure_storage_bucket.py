from django.core.management.base import BaseCommand
from django.conf import settings


class Command(BaseCommand):
    help = 'Ensure the configured S3/MinIO storage bucket exists'

    def handle(self, *args, **options):
        if not getattr(settings, 'AWS_STORAGE_BUCKET_NAME', ''):
            self.stdout.write('S3/MinIO bucket not configured; skipping.')
            return

        bucket = settings.AWS_STORAGE_BUCKET_NAME

        endpoint = getattr(settings, 'AWS_S3_ENDPOINT_URL', None)
        try:
            import boto3
            from botocore.exceptions import ClientError

            client = boto3.client(
                's3',
                endpoint_url=endpoint,
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=getattr(settings, 'AWS_S3_REGION_NAME', None) or 'us-east-1',
            )
            existing = [b['Name'] for b in client.list_buckets().get('Buckets', [])]
            if bucket in existing:
                self.stdout.write(self.style.SUCCESS(f'Bucket already exists: {bucket}'))
                return
            client.create_bucket(Bucket=bucket)
            self.stdout.write(self.style.SUCCESS(f'Created bucket: {bucket}'))
        except ClientError as exc:
            self.stderr.write(self.style.ERROR(f'Failed to ensure bucket: {exc}'))
            raise
        except Exception as exc:
            self.stderr.write(self.style.ERROR(f'Failed to ensure bucket: {exc}'))
            raise
