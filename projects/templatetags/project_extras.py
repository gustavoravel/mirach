from django import template

register = template.Library()

ACTION_LABELS = {
    'project_create': 'Projeto criado',
    'project_update': 'Projeto atualizado',
    'project_delete': 'Projeto excluído',
    'dataset_upload': 'Dataset enviado',
    'dataset_process': 'Dataset processado',
    'dataset_delete': 'Dataset excluído',
    'prediction_create': 'Previsão criada',
    'prediction_run': 'Previsão executada',
    'prediction_delete': 'Previsão excluída',
    'invite_create': 'Convite enviado',
    'invite_accept': 'Convite aceito',
}

@register.filter
def action_label(action: str) -> str:
    return ACTION_LABELS.get(action, action.replace('_', ' ').capitalize())


