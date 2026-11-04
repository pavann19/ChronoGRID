"""Maturity-aware evidence reporting; never mixes experiment kinds or arms."""
from collections import defaultdict
from datetime import datetime, timedelta
import math
import numpy as np
from src.evaluation.prospective import utc
from src.operations.contracts import Forecast, Run
from src.operations.capture import delivery_grid

FIELDS=('point_n','abs_error','squared_error','interval_n','hits','width_sum','winkler_sum','finite_interval_n','unbounded_n','sim_n','sim_hits')
def summarize(v):
    x=dict(zip(FIELDS,np.asarray(v,dtype=float).tolist()))
    def ratio(a,b): return x[a]/x[b] if x[b] else None
    return dict(point_observations=int(x['point_n']),mae_mw=ratio('abs_error','point_n'),rmse_mw=math.sqrt(ratio('squared_error','point_n')) if x['point_n'] else None,interval_observations=int(x['interval_n']),covered=int(x['hits']),missed=int(x['interval_n']-x['hits']),marginal_coverage=ratio('hits','interval_n'),mean_interval_width_mw=None if x['unbounded_n'] else ratio('width_sum','finite_interval_n'),mean_winkler_score=None if x['unbounded_n'] else ratio('winkler_sum','finite_interval_n'),unbounded_intervals=int(x['unbounded_n']),simultaneous_observations=int(x['sim_n']),simultaneous_covered=int(x['sim_hits']),simultaneous_coverage=ratio('sim_hits','sim_n'))

def score_run(store,run_id,cutoff: datetime,replicates=10000):
    cutoff=utc(cutoff)
    if cutoff>utc(store.clock()): raise ValueError('report cutoff in future')
    if not 100<=replicates<=10000: raise ValueError('invalid bootstrap replicate budget')
    integrity=store.verify(); run=Run.model_validate(store.get('run:'+run_id))
    forecasts={}; actuals=defaultdict(list)
    for row in store.rows('forecast'):
        p=Forecast.model_validate(row['payload'])
        if p.run_id==run_id and utc(datetime.fromisoformat(row['recorded_at']))<=cutoff: forecasts[(p.delivery_date,p.arm)]=(p,row)
    for row in store.rows('actual'):
        if utc(datetime.fromisoformat(row['recorded_at']))<=cutoff:
            a=row['payload']; actuals[(a['prediction_sha256'],utc(datetime.fromisoformat(a['delivery_timestamp'])))].append(row)
    results={}
    for arm in run.arms:
        vectors=[]; bootstrap_vectors=[]; origins=[]; strata=defaultdict(lambda:np.zeros(len(FIELDS)))
        scheduled_hours=missing_predictions=missing_actuals=not_mature=unavailable_intervals=0
        day=run.start_date
        while day<=run.end_date:
            grid=delivery_grid(day); v=np.zeros(len(FIELDS)); counts=defaultdict(int)
            found=forecasts.get((day,arm)); p,meta=found if found else (None,None)
            covered=0; eligible_intervals=0; all_mature=True
            for i,t in enumerate(grid):
                scheduled_hours+=1
                maturity=t+timedelta(hours=1,days=run.maturity_days)
                if maturity>cutoff:
                    not_mature+=1; all_mature=False; continue
                if p is None: missing_predictions+=1; continue
                vintage=[a for a in actuals.get((meta['sha256'],t),[]) if utc(datetime.fromisoformat(a['recorded_at']))<=maturity and utc(datetime.fromisoformat(a['payload']['available_at']))<=maturity]
                vintage.sort(key=lambda a:(utc(datetime.fromisoformat(a['recorded_at'])),a['key']))
                # First collected vintage is fixed, even if later revision has better labels.
                actual=vintage[0]['payload'] if vintage else None
                if not actual or actual['status']!='observed': missing_actuals+=1; continue
                y=actual['value_mw']; d=p.deliveries[i]; e=y-d.point_mw
                hour=np.zeros(len(FIELDS)); hour[:3]=[1,abs(e),e*e]
                if d.interval_status=='unavailable': unavailable_intervals+=1
                else:
                    hit=d.interval_status=='unbounded' or d.lower_mw<=y<=d.upper_mw
                    hour[3:5]=[1,int(hit)]; eligible_intervals+=1; covered+=int(hit)
                    if d.interval_status=='unbounded': hour[8]=1
                    else:
                        width=d.upper_mw-d.lower_mw
                        winkler=width+2/(1-p.nominal_coverage)*(max(d.lower_mw-y,0)+max(y-d.upper_mw,0))
                        hour[5:8]=[width,winkler,1]
                v+=hour
                season=('DJF' if day.month in (12,1,2) else 'MAM' if day.month in (3,4,5) else 'JJA' if day.month in (6,7,8) else 'SON')
                regime='low' if y<run.demand_low_mw else 'high' if y>=run.demand_high_mw else 'medium'
                for key in ('season:'+season,'demand:'+regime,'route:'+p.route,'quality:'+('flagged' if p.data_quality_flags or d.quality_flags or actual['quality_flags'] else 'unflagged'),'outage:'+str(p.outage).lower()): strata[key]+=hour
            complete=all_mature and eligible_intervals==len(grid)
            v[9:11]=[int(complete),int(complete and covered==len(grid))]
            origins.append(dict(delivery_date=str(day),hours=len(grid),forecast_present=p is not None,interval_eligible_hours=eligible_intervals,daily_coverage=covered/eligible_intervals if eligible_intervals else None,complete=complete,**summarize(v)))
            vectors.append(v)
            if grid[-1]+timedelta(hours=1,days=run.maturity_days)<=cutoff: bootstrap_vectors.append(v)
            day+=timedelta(days=1)
        overall=summarize(np.asarray(vectors).sum(axis=0)); arr=np.asarray(bootstrap_vectors); ci=[]
        # Every scheduled calendar day is retained; missing days contribute zero counts,
        # so blocks never silently join non-adjacent available days.
        complete_count=sum(x['complete'] for x in origins)
        point_complete=sum(x['point_observations']==x['hours'] for x in origins)
        rng=np.random.RandomState(42)
        for b in (1,2,3,5,7,14):
            if b>5 and complete_count<28 and point_complete<28: continue
            if len(arr)<b: continue
            if max(complete_count,point_complete)<20:
                ci.append(dict(block_days=b,status='insufficient_complete_origins',minimum=20)); continue
            starts=rng.randint(0,len(arr),size=(replicates,int(np.ceil(len(arr)/b))))
            idx=((starts[:,:,None]+np.arange(b))%len(arr)).reshape(replicates,-1)[:,:len(arr)]
            sums=arr[idx].sum(axis=1)
            for key,numerator,denominator,root in [('marginal_coverage',4,3,False),('simultaneous_coverage',10,9,False),('mae_mw',1,0,False),('rmse_mw',2,0,True),('mean_interval_width_mw',5,7,False),('mean_winkler_score',6,7,False)]:
                needed=point_complete if key in ('mae_mw','rmse_mw') else complete_count
                valid=sums[:,denominator]>0
                if needed<20 or not valid.all() or (key in ('mean_interval_width_mw','mean_winkler_score') and sums[:,8].any()):
                    ci.append(dict(block_days=b,statistic=key,status='insufficient_or_unbounded',valid_replicates=int(valid.sum()))); continue
                samples=sums[:,numerator]/sums[:,denominator]
                if root: samples=np.sqrt(samples)
                low,high=np.percentile(samples,[2.5,97.5])
                ci.append(dict(block_days=b,statistic=key,status='empirical',ci_low=float(low),ci_high=float(high),std_err=float(samples.std()),degenerate=bool(low==high),replicates=replicates))
        daily=[x['daily_coverage'] for x in origins if x['complete']]
        results[arm]=dict(overall=overall,scheduled_origins=len(origins),scheduled_hours=scheduled_hours,missing_forecast_hours=missing_predictions,missing_actual_hours=missing_actuals,not_mature_hours=not_mature,unavailable_interval_hours=unavailable_intervals,complete_origins=complete_count,bootstrap_calendar_origins=len(bootstrap_vectors),bootstrap_scope='fully_mature_scheduled_calendar_days_only; missing past days retained',minimum_complete_daily_coverage=min(daily) if daily else None,daily_coverage_quantiles=dict(zip(('p05','p50','p95'),np.percentile(daily,[5,50,95]).tolist())) if daily else None,worst_origins=sorted((x for x in origins if x['daily_coverage'] is not None),key=lambda x:(x['daily_coverage'],x['delivery_date']))[:10],per_origin=origins,strata={k:summarize(v) for k,v in sorted(strata.items())},bootstrap=ci)
    return dict(schema_version='1.0',run_id=run_id,experiment_kind=run.experiment_kind,arm_identity_policy='per_arm_model_pin' if run.arm_models else 'legacy_run_allowed_model_set_only',report_cutoff=cutoff.isoformat(),actual_policy='first_collected_vintage_by_7_day_maturity' if run.maturity_days==7 else f'first_collected_vintage_by_{run.maturity_days}_day_maturity',integrity=integrity,arms=results,limitations=['Local commitment is not an independent timestamp receipt.','Calendar days including missing days are resampled; dependence beyond tested block lengths remains unknown.','Intervals and strata are empirical, not guarantees; unbounded widths are null with explicit counts.','No counterfactual head comparison without genuinely issued paired records.'])
