"""Read-only Parquet/XLSX parity using independent openpyxl workbook reader."""
import argparse
from pathlib import Path
import json
import hashlib
from datetime import date,datetime
import time

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'private_runs/tail240_20260916/forensics/atfm'
TABLES=BASE/'inspection_v1/tables'
OUT=ROOT/'private_runs/tail240_20260916/validation/atfm_workbooks_v2'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):
            h.update(block)
    return h.hexdigest()


def prepare():
    import numpy as np
    import pandas as pd
    manifest=json.loads((TABLES/'manifest.json').read_text())
    assert manifest['status']=='complete'
    for name,digest in manifest['outputs'].items():
        assert sha(TABLES/name)==digest
    OUT.mkdir(parents=True,exist_ok=False)
    reports=[]
    for report in manifest['reports']:
        path=TABLES/(Path(report['file']).stem+'_airports.parquet')
        table=pd.read_parquet(path)
        assert len(table)==report['retained_rows'] and not table.duplicated(['APT_ICAO','date_utc']).any()
        assert all(not item['missing_dates'] and item['days']==item['expected_days'] for item in report['coverage'])
        samples=[]
        for _,group in table.groupby('APT_ICAO'):
            relevant=group.loc[group.date_utc.str.startswith(('2025','2026-01','2026-07'))]
            pick=np.unique(np.linspace(0,len(relevant)-1,5,dtype=int))
            samples.extend(relevant.iloc[pick].source_row.astype(int).tolist())
        numeric=[c for c in report['headers'] if c.startswith(('DLY_','FLT_DEP','FLT_ARR'))]
        for column in numeric:
            for mask in (table[column].isna(),table[column].eq(0),table[column].gt(0)):
                subset=table.loc[mask]
                if len(subset):
                    samples.append(int(subset.iloc[0].source_row))
        sample=table.loc[table.source_row.isin(samples)].sort_values('source_row')
        checks={}
        if 'FLT_DEP_REG_1' in table:
            sumcounts=table.FLT_DEP_OUT_EARLY_1+table.FLT_DEP_IN_1+table.FLT_DEP_OUT_LATE_1
            assert np.array_equal(sumcounts,table.FLT_DEP_REG_1)
            checks['slot_component_sum_exact_rows']=len(table)
        else:
            delay=[c for c in table if c.startswith('DLY_')]
            checks['all_delay_fields_blank_rows']=int(table[delay].isna().all(axis=1).sum())
            checks['true_zero_total_delay_rows']=int(table.DLY_APT_ARR_1.eq(0).sum())
        records=json.loads(sample.to_json(orient='records'))
        reports.append({'file':report['file'],'source_sha256':report['sha256'],'headers':report['headers'],
            'retained_rows':len(table),'samples':records,'table_checks':checks})
    (OUT/'prepared_samples.json').write_text(json.dumps({'source_sha256':sha(__file__),'producer_manifest_sha256':sha(TABLES/'manifest.json'),'reports':reports},indent=2))
    print('PREPARED',[(r['file'],len(r['samples']),r['table_checks']) for r in reports],flush=True)


def verify():
    import openpyxl
    from openpyxl.utils.datetime import to_excel
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(name,ctypes.c_size_t) for name in
            ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
             'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    psapi=ctypes.WinDLL('psapi',use_last_error=True)
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    def rss():
        result=Counters()
        result.cb=ctypes.sizeof(result)
        assert psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(result),result.cb)
        return result.WorkingSetSize
    prepared=json.loads((OUT/'prepared_samples.json').read_text())
    assert prepared['source_sha256']==sha(__file__) and prepared['producer_manifest_sha256']==sha(TABLES/'manifest.json')
    reports=[]
    began=time.monotonic()
    for source in prepared['reports']:
        path=BASE/'acquisition_v1'/source['file']
        assert sha(path)==source['source_sha256']
        workbook=openpyxl.load_workbook(path,read_only=True,data_only=True)
        sheet=workbook['DATA']
        wanted={int(record['source_row']):record for record in source['samples']}
        found={}
        stream=sheet.iter_rows(min_row=1,max_row=max(wanted),max_col=len(source['headers']))
        first=next(stream)
        headers=[cell.value for cell in first]
        assert headers==source['headers']
        cells,blank,zero=0,0,0
        for rowno,row in enumerate(stream,start=2):
            if rowno%200000==0:
                assert rss()<4*1024**3
                print('XLSX_SCANNED',source['file'],rowno,flush=True)
            if rowno not in wanted:
                continue
            expected=wanted[rowno]
            locations=[]
            for header,cell in zip(headers,row):
                actual=cell.value
                reference=expected.get(header)
                if header=='FLT_DATE':
                    actual_date=actual.date().isoformat() if isinstance(actual,datetime) else actual.isoformat() if isinstance(actual,date) else openpyxl.utils.datetime.from_excel(actual,workbook.epoch).date().isoformat()
                    assert actual_date==expected['date_utc']
                    if isinstance(actual,(date,datetime)):
                        actual=to_excel(actual,workbook.epoch)
                if reference is None:
                    assert actual is None
                    blank+=1
                elif isinstance(reference,(int,float)):
                    assert isinstance(actual,(int,float)) and abs(actual-reference)<=1e-8*max(1,abs(reference))
                    zero+=int(actual==0)
                else:
                    assert actual==reference,(source['file'],rowno,header,actual,reference)
                cells+=1
                locations.append(cell.coordinate if hasattr(cell,'coordinate') else None)
            found[rowno]=locations
        assert set(found)==set(wanted)
        workbook.close()
        assert sha(path)==source['source_sha256']
        reports.append({'file':source['file'],'sample_rows':len(found),'cells_checked':cells,'blank_cells_preserved':blank,
            'finite_zero_cells_preserved':zero,'source_rows_and_cell_coordinates':found,'table_checks':source['table_checks']})
        print('XLSX_VERIFIED',source['file'],len(found),cells,flush=True)
    receipt={'status':'passed','source_sha256':sha(__file__),'sample_definition_sha256':sha(OUT/'prepared_samples.json'),
        'producer_manifest_sha256':prepared['producer_manifest_sha256'],'openpyxl_version':openpyxl.__version__,
        'reports':reports,'runtime_sec':time.monotonic()-began,'rss_bytes':rss(),
        'limits':'Original workbook bytes unchanged. Independent read-only openpyxl reader checks sampled row/cellvalues, truezeros, sparseblanks andExceldateconversion; alltablecomponent/coverage checks useextractedtables. Source dates say Dateofflight; UTCsemantics not established. No privateflightlabels/blockclocks loaded.'}
    (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2))
    print('COMPLETE',sum(r['cells_checked'] for r in reports),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    prepare() if args.prepare else verify()
