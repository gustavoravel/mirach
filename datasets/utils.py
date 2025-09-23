import pandas as pd
import numpy as np
from django.core.files.uploadedfile import UploadedFile
import os
from typing import Dict, List, Tuple, Optional


def process_file(file: UploadedFile) -> Dict:
    """
    Processa um arquivo (Excel ou CSV) e retorna informações sobre o dataset
    """
    try:
        ext = os.path.splitext(file.name)[1].lower()
        detected = {'encoding': None, 'delimiter': None}
        # Ler arquivo conforme extensão
        if ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file)
        elif ext == '.csv':
            # Tentar detecção simples de encoding e delimitador
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
            last_err = None
            for enc in encodings:
                try:
                    file.seek(0)
                    df = pd.read_csv(file, sep=None, engine='python', encoding=enc, low_memory=False)
                    detected['encoding'] = enc
                    # Tentar inferir separador do primeiro row/columns length
                    detected['delimiter'] = getattr(df, 'attrs', {}).get('delimiter', None)
                    break
                except Exception as e:
                    last_err = e
                    continue
            else:
                raise last_err or ValueError('Falha ao ler CSV com encodings comuns')
        else:
            raise ValueError(f"Formato não suportado: {ext}")
        
        # Informações básicas
        total_rows = len(df)
        total_columns = len(df.columns)
        column_names = df.columns.tolist()
        
        # Detectar tipos de dados
        column_types = {}
        for col in df.columns:
            if df[col].dtype == 'object':
                # Tentar converter para datetime
                try:
                    pd.to_datetime(df[col], errors='raise')
                    column_types[col] = 'datetime'
                except:
                    column_types[col] = 'text'
            elif df[col].dtype in ['int64', 'int32']:
                column_types[col] = 'integer'
            elif df[col].dtype in ['float64', 'float32']:
                column_types[col] = 'float'
            else:
                column_types[col] = 'other'
        
        # Estatísticas básicas
        stats = {}
        for col in df.columns:
            if column_types[col] in ['integer', 'float']:
                stats[col] = {
                    'mean': float(df[col].mean()) if not df[col].isna().all() else None,
                    'std': float(df[col].std()) if not df[col].isna().all() else None,
                    'min': float(df[col].min()) if not df[col].isna().all() else None,
                    'max': float(df[col].max()) if not df[col].isna().all() else None,
                    'null_count': int(df[col].isna().sum())
                }
        
        return {
            'success': True,
            'total_rows': total_rows,
            'total_columns': total_columns,
            'column_names': column_names,
            'column_types': column_types,
            'statistics': stats,
            'dataframe': df,
            'detected': detected
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


# Backward compatibility
def process_excel_file(file: UploadedFile) -> Dict:
    return process_file(file)


def detect_time_series_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Detecta automaticamente colunas de timestamp e variável alvo
    """
    suggestions = {}
    
    for col in df.columns:
        # Detectar colunas de timestamp
        if df[col].dtype == 'object':
            try:
                pd.to_datetime(df[col], errors='raise')
                suggestions[col] = 'timestamp'
            except:
                pass
        
        # Detectar colunas numéricas como possíveis variáveis alvo
        elif df[col].dtype in ['int64', 'int32', 'float64', 'float32']:
            # Se tem poucos valores únicos, pode ser categórico
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.1:  # Mais de 10% de valores únicos
                suggestions[col] = 'target'
            else:
                suggestions[col] = 'feature'
        else:
            suggestions[col] = 'ignore'
    
    return suggestions


def validate_time_series_data(df: pd.DataFrame, timestamp_col: str, target_col: str) -> Dict:
    """
    Valida se os dados são adequados para análise de séries temporais
    """
    errors = []
    warnings = []
    
    # Verificar se as colunas existem
    if timestamp_col not in df.columns:
        errors.append(f"Coluna de timestamp '{timestamp_col}' não encontrada")
    
    if target_col not in df.columns:
        errors.append(f"Coluna alvo '{target_col}' não encontrada")
    
    if errors:
        return {'valid': False, 'errors': errors, 'warnings': warnings}
    
    # Converter timestamp
    try:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    except:
        errors.append(f"Não foi possível converter '{timestamp_col}' para datetime")
        return {'valid': False, 'errors': errors, 'warnings': warnings}
    
    # Verificar se target é numérico
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        errors.append(f"Coluna alvo '{target_col}' deve ser numérica")
    
    # Verificar valores nulos
    null_timestamp = df[timestamp_col].isna().sum()
    null_target = df[target_col].isna().sum()
    
    if null_timestamp > 0:
        warnings.append(f"Encontrados {null_timestamp} valores nulos na coluna de timestamp")
    
    if null_target > 0:
        warnings.append(f"Encontrados {null_target} valores nulos na coluna alvo")
    
    # Verificar duplicatas de timestamp
    duplicates = df[timestamp_col].duplicated().sum()
    if duplicates > 0:
        warnings.append(f"Encontrados {duplicates} timestamps duplicados")
    
    # Verificar se os dados estão ordenados por tempo
    is_sorted = df[timestamp_col].is_monotonic_increasing
    if not is_sorted:
        warnings.append("Os dados não estão ordenados cronologicamente")
    
    # Detecção simples de outliers no alvo (Z-score e IQR)
    try:
        series = pd.to_numeric(df[target_col], errors='coerce')
        series = series.dropna()
        if len(series) > 0:
            mean = series.mean()
            std = series.std()
            if std and std > 0:
                z_scores = ((series - mean).abs() / std)
                z_outliers = int((z_scores > 3).sum())
                if z_outliers > 0:
                    warnings.append(f"Detectados {z_outliers} outliers pelo critério Z-score > 3")
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if iqr and iqr > 0:
                iqr_outliers = int(((series < (q1 - 1.5*iqr)) | (series > (q3 + 1.5*iqr))).sum())
                if iqr_outliers > 0:
                    warnings.append(f"Detectados {iqr_outliers} outliers pelo critério IQR")
    except Exception:
        pass

    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }


def prepare_time_series_data(df: pd.DataFrame, timestamp_col: str, target_col: str, 
                            feature_cols: List[str] = None) -> pd.DataFrame:
    """
    Prepara os dados para análise de séries temporais
    """
    # Converter timestamp
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Ordenar por timestamp
    df = df.sort_values(timestamp_col)
    
    # Remover duplicatas de timestamp (manter o primeiro)
    df = df.drop_duplicates(subset=[timestamp_col], keep='first')
    
    # Selecionar colunas relevantes
    cols_to_keep = [timestamp_col, target_col]
    if feature_cols:
        cols_to_keep.extend(feature_cols)
    
    df_clean = df[cols_to_keep].copy()
    
    # Definir timestamp como índice
    df_clean = df_clean.set_index(timestamp_col)
    
    return df_clean
