import pandas as pd 
import numpy as np
import os
import pandas as pd
from datetime import date
import geopandas
import gc

#Read DC311 data / sidewalks
d311 = pd.concat([pd.read_csv(f'./Data/{x}') for x in os.listdir('./Data') if '311' in x])
sidewalks = d311.query("SERVICECODE == 'S0361'")\
    .query("WARD >= 1 and WARD <= 8")

#Normalize dates
for col in [x for x in sidewalks.columns if 'DATE' in x]:
    sidewalks[col] = pd.to_datetime(sidewalks[col])

#Fix priority delination
sidewalks['PRIORITY'] = sidewalks['PRIORITY'].str.upper()
sidewalks['PRIORITY'] = sidewalks['PRIORITY'].str.replace("EMERGNCY","EMERGENCY")
sidewalks['SERVICEORDERSTATUS'] = sidewalks['SERVICEORDERSTATUS'].str.upper()

sidewalks['SLA_DUE_BIZ_DAYS'] = np.busday_count(sidewalks['ADDDATE'].values.astype('datetime64[D]'),sidewalks['SERVICEDUEDATE'].values.astype('datetime64[D]'))
sidewalks['COMPLETE_DAYS'] = sidewalks.apply(lambda row: np.busday_count(row['ADDDATE'].date(),row['RESOLUTIONDATE'].date()) if (pd.notnull(row['ADDDATE']) and pd.notnull(row['RESOLUTIONDATE']) and row['SERVICEORDERSTATUS'] == 'CLOSED') else np.nan,axis=1)
sidewalks['DAYS_TO_TODAY'] = sidewalks.apply(lambda row: np.busday_count(row['ADDDATE'].date(),date.today()) if (pd.notnull(row['ADDDATE'])) else np.nan,axis=1)

#Calculate KM event timeline
sidewalks['EVENT_DAYS'] = np.where(sidewalks['SERVICEORDERSTATUS'] == 'CLOSED',sidewalks['COMPLETE_DAYS'],sidewalks['DAYS_TO_TODAY'])
sidewalks['EVENT_CLOSED'] = np.where(sidewalks['SERVICEORDERSTATUS'] == 'CLOSED',1,0)

#Pull in City Works Requests / Work Orders
creq = pd.read_csv('./Data/Cityworks_Service_Requests.csv')
cwo = pd.read_csv("./Data/Cityworks_Workorders.csv")

#Normalize dates
for col in [x for x in creq.columns if 'DATE' in x]:
    creq[col] = pd.to_datetime(creq[col],errors='ignore')

for col in [x for x in cwo.columns if 'DATE' in x]:
    cwo[col] = pd.to_datetime(cwo[col],errors='ignore')   

#Pull relevant CityWorks request data
creq_filt = creq[['REQUESTID','WORKORDERID','CSRNUMBER','STATUS','REQUESTCATEGORY', 'INITIATEDDATE',
       'CLOSEDDATE','DESCRIPTION', 'INSPECTIONDATE', 'INSPECTIONCOMPLETE', 'SUBMITTEDTODATE',
       'DISPATCHEDTODATE', 'CANCELEDDATE', 'PRIORITY', 'INITIATEDBY',
       'SUBMITTEDTO', 'DISPATCHEDTO', 'CLOSEDBY', 'PROJECTNAME', 'ISCANCELED',
       'CANCELEDBY','DAYSTOCLOSE', 'DAYSTOINSPECT','X','Y']]
creq_filt = creq_filt.rename(columns={x:'CW_REQ_'+x for x in creq_filt.columns})

#Pull relevant CityWorks work order data
cwo_filt = cwo[['WORKORDERID', 'PROJECTID', 'DESCRIPTION',
       'STATUS', 'INITIATEDDATE', 'WORKORDERCLOSEDDATE', 'ACTUALSTARTDATE',
       'ACTUALFINISHDATE', 'PROJECTNAME', 'PRIORITY', 'SOURCEWORKORDERID',
       'CYCLETYPE', 'SCHEDULEDATE', 'WORKORDERCATEGORY', 'UNATTACHED',
       'WORKORDERCOST', 'WORKORDERLABORCOST', 'WORKORDERMATERIALCOST',
       'WORKORDEREQUIPMENTCOST', 'SUBMITTEDTO', 'SUBMITTEDTODATE',
       'WORKCOMPLETEDBY', 'WORKORDERCLOSEDBY', 'ISCANCELED', 'CANCELEDBY',
       'CANCELEDDATE', 'ASSETGROUP', 'SUPERVISOR', 'REQUESTEDBY','X','Y']]
cwo_filt = cwo_filt.rename(columns={x:'CW_WO_'+x for x in cwo_filt.columns})

#Pull in Census GEOID to join
c2020 = geopandas.read_file('./Data/Census_Blocks_in_2020.geojson')
c2020 = c2020[['OBJECTID','BLKGRP','BLOCK','GEOID','P0010001','geometry']]\
    .rename(columns={'P0010001':'TOTAL_POP'})

#Map data elements into geo pandas to pull block groups
swalk_geo = geopandas.GeoDataFrame(sidewalks[['X','Y','SERVICEREQUESTID']], geometry=geopandas.points_from_xy(sidewalks['X'],sidewalks['Y'])).set_crs('EPSG:4326')
cwo_geo = geopandas.GeoDataFrame(cwo_filt[['CW_WO_X','CW_WO_Y','CW_WO_WORKORDERID']], geometry=geopandas.points_from_xy(cwo_filt['CW_WO_X'],cwo_filt['CW_WO_Y'])).set_crs('EPSG:4326')
creq_geo = geopandas.GeoDataFrame(creq_filt[['CW_REQ_X','CW_REQ_Y','CW_REQ_REQUESTID']], geometry=geopandas.points_from_xy(creq_filt['CW_REQ_X'],creq_filt['CW_REQ_Y'])).set_crs('EPSG:4326')

#Rejoin block to request
sidewalks = sidewalks\
    .merge(swalk_geo.sjoin(c2020, predicate="within")[['SERVICEREQUESTID','GEOID','TOTAL_POP']],how='left',on='SERVICEREQUESTID')

cwo_filt = cwo_filt\
    .merge(cwo_geo.sjoin(c2020, predicate="within")[['CW_WO_WORKORDERID','GEOID','TOTAL_POP']],how='left',on='CW_WO_WORKORDERID')

creq_filt = creq_filt\
    .merge(creq_geo.sjoin(c2020,predicate="within")[['CW_REQ_REQUESTID','GEOID','TOTAL_POP']],how='left',on='CW_REQ_REQUESTID')

#Examine sidewalks that join to get general descriptive stats
t_sidewalks = sidewalks.merge(creq_filt[['CW_REQ_REQUESTID','CW_REQ_INITIATEDDATE','CW_REQ_DESCRIPTION','CW_REQ_CSRNUMBER']],how='inner',left_on='SERVICEREQUESTID',right_on='CW_REQ_CSRNUMBER')
#Vast majority are SIDEWALK REPAIR
print(t_sidewalks['CW_REQ_DESCRIPTION'].value_counts(dropna=False))
print((t_sidewalks['CW_REQ_INITIATEDDATE'] - t_sidewalks['ADDDATE']).describe())


#Free up some memory hopefully
del d311,creq,cwo,t_sidewalks
del cwo_geo,swalk_geo,creq_geo
gc.collect()

#Filter on sidewalks
creq_filt = creq_filt.query("CW_REQ_DESCRIPTION == 'SIDEWALK REPAIR'")
cwo_filt = cwo_filt[cwo_filt['CW_WO_DESCRIPTION'].str.contains('SIDEWALK',na=False)]
gc.collect()

#Map based on location and time
nm_sidewalks = sidewalks[['GEOID','SERVICEREQUESTID','ADDDATE']].drop_duplicates()\
    .merge(creq_filt[['GEOID','CW_REQ_INITIATEDDATE','CW_REQ_REQUESTID']],how='inner',on='GEOID')
nm_sidewalks['DATE_DIFF'] = np.abs((nm_sidewalks['CW_REQ_INITIATEDDATE'] - nm_sidewalks['ADDDATE']).dt.days)
nm_sidewalks = nm_sidewalks.query("DATE_DIFF <= 1").drop_duplicates(subset=['SERVICEREQUESTID','CW_REQ_REQUESTID'])

s_to_req = nm_sidewalks.merge(creq_filt[['CW_REQ_REQUESTID','CW_REQ_WORKORDERID','CW_REQ_CSRNUMBER','CW_REQ_STATUS', 'CW_REQ_CLOSEDDATE']],how='inner',on='CW_REQ_REQUESTID')
s_to_req['REQ_MATCH'] = np.where(s_to_req['SERVICEREQUESTID'] == s_to_req['CW_REQ_CSRNUMBER'],'DIRECT','TANDD')


req_to_wo = s_to_req[['CW_REQ_REQUESTID','GEOID','CW_REQ_INITIATEDDATE','CW_REQ_WORKORDERID','CW_REQ_CLOSEDDATE']]\
    .drop_duplicates()\
    .merge(cwo_filt[['CW_WO_WORKORDERID','GEOID','CW_WO_STATUS','CW_WO_INITIATEDDATE','CW_WO_WORKORDERCLOSEDDATE','CW_WO_ACTUALFINISHDATE']],how='left',on='GEOID')

#Filter to a work order created in the period or a direct tie
req_to_wo['FILTER_DATE'] = np.where((req_to_wo['CW_REQ_WORKORDERID'].astype('float64') == req_to_wo['CW_WO_WORKORDERID']) | ((req_to_wo['CW_WO_INITIATEDDATE'] >= req_to_wo['CW_REQ_INITIATEDDATE']) & (req_to_wo['CW_WO_INITIATEDDATE'] <= req_to_wo['CW_REQ_CLOSEDDATE'])),True,False)
req_to_wo = req_to_wo[req_to_wo['FILTER_DATE']]

#How it's matched
req_to_wo['REQ_MATCH'] = np.where(req_to_wo['CW_REQ_WORKORDERID'] == req_to_wo['CW_WO_WORKORDERID'],'DIRECT','TANDD')
req_to_wo = req_to_wo.join(pd.get_dummies(req_to_wo['CW_WO_STATUS'],prefix='WO'))

#Roll ups
req_to_wo['CLOSED_DIRECT'] = np.where((req_to_wo['CW_WO_STATUS'] == 'CLOSED') & (req_to_wo['REQ_MATCH'] == 'DIRECT'),1,0)
req_to_wo['CLOSED_TM'] = np.where((req_to_wo['CW_WO_STATUS'] == 'CLOSED') & (req_to_wo['REQ_MATCH'] != 'DIRECT'),1,0)
req_to_wo['WO_INIT'] = 1

wo_roll = req_to_wo.pivot_table(index='CW_REQ_REQUESTID',values=['WO_CLOSED','WO_OPEN','WO_PENDING','WO_SCHEDULED','WO_INIT','CW_WO_WORKORDERCLOSEDDATE'],aggfunc={'WO_CLOSED':np.sum,'WO_OPEN':np.sum,'WO_PENDING':np.sum,'WO_SCHEDULED':np.sum,'WO_INIT':np.sum,'CW_WO_WORKORDERCLOSEDDATE':np.max})\
    .rename(columns={'CW_WO_WORKORDERCLOSEDDATE':'WO_MAX_CLOSE'})\
    .reset_index()

s_to_req['REQ_INIT'] = 1
s_to_req = s_to_req.join(pd.get_dummies(s_to_req['CW_REQ_STATUS'],prefix='REQ'))

req_wo = s_to_req\
    .merge(wo_roll,how='left',on='CW_REQ_REQUESTID')\
    .pivot_table(index='SERVICEREQUESTID',
    values=['CW_REQ_CLOSEDDATE','REQ_CLOSED', 'REQ_COMPLETE', 'REQ_INSPCOMP', 'REQ_OPEN', 'REQ_PENDING',
       'REQ_INIT', 'WO_MAX_CLOSE', 'WO_CLOSED', 'WO_INIT', 'WO_OPEN',
       'WO_PENDING', 'WO_SCHEDULED'], aggfunc={'CW_REQ_CLOSEDDATE':np.max,'REQ_CLOSED':np.sum, 'REQ_COMPLETE':np.sum, 'REQ_INSPCOMP':np.sum, 'REQ_OPEN':np.sum, 'REQ_PENDING':np.sum,
       'REQ_INIT':np.sum, 'WO_MAX_CLOSE':np.max, 'WO_CLOSED':np.sum, 'WO_INIT':np.sum, 'WO_OPEN':np.sum,'WO_PENDING':np.sum, 'WO_SCHEDULED':np.sum})\
    .reset_index()\
    .rename(columns={'CW_REQ_CLOSEDDATE':'REQ_MAX_CLOSE_DATE'})

#Merge back
sidewalks_comb = sidewalks\
    .merge(req_wo,how='left',on='SERVICEREQUESTID')

int_cols = [x for x in sidewalks_comb.columns if ('REQ_' in x or 'WO_' in x) and 'MAX' not in x]
sidewalks_comb[int_cols] = sidewalks_comb[int_cols].fillna(0)

#Closed without a WO
sidewalks_comb['D311_CLOSED_NO_WO'] = np.where((sidewalks_comb['SERVICEORDERSTATUS'] == 'CLOSED') & (sidewalks_comb['WO_CLOSED'] == 0),1,0)

#Days from Close to Add Date
sidewalks_comb['WO_CLOSE_FROM_D311_ADD'] = (sidewalks_comb['WO_MAX_CLOSE'] - sidewalks['ADDDATE']).dt.days
sidewalks_comb['D311_CLOSE_FROM_D311_ADD'] = (sidewalks_comb['RESOLUTIONDATE'] - sidewalks['ADDDATE']).dt.days

#Calculate KM closed date
sidewalks_comb['WO_COMPLETE_BIZ_DAYS'] = sidewalks_comb.apply(lambda row: np.busday_count(row['ADDDATE'].date(),row['WO_MAX_CLOSE'].date()) if (pd.notnull(row['ADDDATE']) and pd.notnull(row['WO_MAX_CLOSE']) and row['WO_CLOSED'] >= 1) else np.nan,axis=1)

#Calculate KM event timeline for WO CLOSE
sidewalks_comb['WO_EVENT_DAYS'] = np.where(sidewalks_comb['WO_CLOSED'] >= 1,sidewalks_comb['WO_COMPLETE_BIZ_DAYS'],sidewalks_comb['DAYS_TO_TODAY'])
sidewalks_comb['WO_EVENT_CLOSED'] = np.where(sidewalks_comb['WO_CLOSED'] >= 1,1,0)

#Additional filter cols
sidewalks_comb['FY'] = sidewalks_comb['ADDDATE'].apply(lambda x: x.year + (1 if x.month >= 10 else 0))

#Cnt
sidewalks_comb['D311_CNT'] = 1

sidewalks_comb['SERVICEORDERSTATUS'] = sidewalks_comb['SERVICEORDERSTATUS'].str.replace('IN-PROGRESS','OPEN')
#Output
sidewalks_comb.to_csv('sidewalks_311.csv',index=False)