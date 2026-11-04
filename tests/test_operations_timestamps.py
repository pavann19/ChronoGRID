"""Actual OpenSSL signing/verification against an ephemeral LOCAL TEST CA only."""
from datetime import datetime,timedelta,timezone
from pathlib import Path
import shutil,subprocess,json
import pytest
from src.operations.timestamps import make_request,verify_receipt
from src.operations.journal import Journal,digest,encode
from src.operations.contracts import Run,Forecast
from test_operations_evidence import forecast,Clock

OPENSSL=shutil.which('openssl')
@pytest.mark.skipif(OPENSSL is None,reason='OpenSSL executable unavailable; RFC3161 integration unverified')
def test_rfc3161_local_test_authority_signature_nonce_and_imprint(tmp_path):
    def openssl(*args):
        result=subprocess.run([OPENSSL,*map(str,args)],cwd=tmp_path,capture_output=True,timeout=30)
        assert result.returncode==0,result.stderr.decode(errors='replace')
    openssl('req','-x509','-newkey','rsa:2048','-nodes','-keyout','root.key','-out','root.pem','-days','2','-subj','/CN=ChronoGrid Local Test Root','-addext','basicConstraints=critical,CA:TRUE','-addext','keyUsage=critical,keyCertSign,cRLSign')
    openssl('req','-new','-newkey','rsa:2048','-nodes','-keyout','tsa.key','-out','tsa.csr','-subj','/CN=ChronoGrid LOCAL TEST TSA')
    (tmp_path/'ext.cnf').write_text('basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature\nextendedKeyUsage=critical,timeStamping\n')
    openssl('x509','-req','-in','tsa.csr','-CA','root.pem','-CAkey','root.key','-CAcreateserial','-out','tsa.pem','-days','2','-extfile','ext.cnf')
    (tmp_path/'serial').write_text('01\n')
    config='''[tsa]
default_tsa=tsa_config
[tsa_config]
serial=serial
signer_cert=tsa.pem
certs=root.pem
signer_key=tsa.key
signer_digest=sha256
default_policy=1.2.3.4.1
other_policies=1.2.3.4.2
digests=sha256
accuracy=secs:1
clock_precision_digits=0
ordering=no
tsa_name=yes
ess_cert_id_chain=no
ess_cert_id_alg=sha256
'''
    (tmp_path/'tsa.cnf').write_text(config)
    now=datetime.now(timezone.utc);day=(now+timedelta(days=2)).date(); clock=Clock();clock.t=now-timedelta(minutes=3)
    store=Journal(tmp_path/'j.sqlite',clock);D='a'*64
    store.register(Run(run_id='run1',experiment_kind='synthetic_fault',protocol_sha256=D,start_date=day,end_date=day,arms=('main',),allowed_model_sha256=(D,),demand_low_mw=90,demand_high_mw=110))
    snapshot=store.snapshot(b'{}','fixture',clock())
    p=forecast(store,clock,snapshot,day=day,kind='synthetic_fault')
    p=Forecast.model_validate(p.model_dump()|{'issued_at':now-timedelta(minutes=1)})
    clock.t=now; h=store.commit(p)
    make_request(h,tmp_path/'request.tsq',OPENSSL)
    openssl('ts','-reply','-config','tsa.cnf','-queryfile','request.tsq','-out','response.tsr')
    clock.t=datetime.now(timezone.utc)+timedelta(seconds=2)
    ca=tmp_path/'root.pem';ca_hash=digest(ca.read_bytes())
    verify_receipt(store,'run1',day,'main',tmp_path/'request.tsq',tmp_path/'response.tsr',ca,ca_hash,'LOCAL TEST','local_test',OPENSSL)
    assert store.rows('receipt')[0]['payload']['trust_scope']=='local_test'
    assert not store.verify()['independent_timestamp']
    make_request('b'*64,tmp_path/'wrong.tsq',OPENSSL)
    with pytest.raises(ValueError): verify_receipt(store,'run1',day,'main',tmp_path/'wrong.tsq',tmp_path/'response.tsr',ca,ca_hash,'LOCAL TEST','local_test',OPENSSL)
    with pytest.raises(ValueError,match='anchor'): verify_receipt(store,'run1',day,'main',tmp_path/'request.tsq',tmp_path/'response.tsr',ca,'b'*64,'LOCAL TEST','local_test',OPENSSL)
