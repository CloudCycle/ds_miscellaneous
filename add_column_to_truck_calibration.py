import awswrangler as wr
import pandas as pd
from decimal import Decimal




def get_all_items_from_table(table_name:str)->pd.DataFrame:
    table = wr.dynamodb.get_table(table_name)
    response = table.scan()
    items = response['Items']
    while 'LastEvaluatedKey' in response:
        print(response['LastEvaluatedKey'])
        response = calibtaion_table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
        items.extend(response['Items'])
        
    return pd.DataFrame(items)
        
    

if __name__ == '__main__':
    
    
    calibration_table_name = 'Production-DataStorage-Stack-CalibrationC17CAA4E-1R3Y188GEBTTK'
    
    truck_table_name = 'Production-DataStorage-Stack-TruckEB9E62B5-189YW1Z3K0370'
        
    df_calibration = get_all_items_from_table(calibration_table_name)
    df_truck = get_all_items_from_table(truck_table_name)

    df_calibration.to_pickle('calibration_table.pkl')
    df_truck.to_pickle('truck_table.pkl')
    
    # Add a project column for all trucks and assign BBV trucks correct project
    df_truck['project'] = '<empty>'
    bbv_trucks = ['FJ22PXL', 'FJ22PXV', 'FJ22PXU', 'FJ22PXY', 'FJ22PXO', 'FJ22PXX'] 
    for i,truck in df_truck.iterrows():
        registration = truck['registration']
        truck_id = truck['truck_id']
        if registration in bbv_trucks:
            df_truck.loc[df_truck['truck_id']==truck_id, 'project'] = 'bbv'
            
    # Save updated table to dynamo (after validating)
    print('-------- Enable Writing after making sure Table is correct ----------')
#     if (len(df_truck) == 70) and (df_truck['truck_id'].unique().shape[0] == 69): # Truck id #2 is repeated
#         wr.dynamodb.put_df(df_truck, truck_table_name)
#     else:
#         print('Inconsistent data in dataframe')
    
    
    # Get all calibrations and map them against a truck for audit
    df_truck_cal = df_truck.merge(df_calibration, how='left', on='truck_id', suffixes=('', '_cal'))
    cols = ['device_imei',
           'registration',
           'motor_displacement_cm3',
           'gearbox_ratio']
    
    df_truck_cal[cols].to_csv('truck_cals.csv')
    
    # Updated cal (export monday table as excel
    column_map = {'Name':'registration', 'Gear Box Ratio': 'gearbox_ratio', 'Motor Displacement': 'motor_displacement_cm3'}
    
    df_updated = (pd
                  .read_excel('Truck_Tracker_1658238281.xlsx', header=2)
                  .rename(column_map, axis=1)
                  .assign(gearbox_ratio = lambda x: [round(Decimal(x),1) if not np.isnan(x) else x for x in x['gearbox_ratio']],
                         motor_displacement_cm3 = lambda x: [round(Decimal(x),1) if not np.isnan(x) else x for x in x['motor_displacement_cm3']])
                 )[column_map.values()]
    
    
    df_truck_cal_updates = df_truck_cal.merge(df_updated,how='left', on='registration', suffixes=('', '_updated'))
    
    # Update cals from Monday to fetched dynamo
    for i,row in df_truck_cal_updates.iterrows():
        registration = row['registration']
        truck_id = row['truck_id']
        gearbox_ratio = row['gearbox_ratio']
        gearbox_ratio_updated = row['gearbox_ratio_updated']
        motor_displacement_cm3 = row['motor_displacement_cm3']
        motor_displacement_cm3_updated = row['motor_displacement_cm3_updated']
        
        if isinstance(gearbox_ratio_updated, Decimal) and (gearbox_ratio != gearbox_ratio_updated):
            print(f"Truck: {registration}, gearbox ratio updated from {gearbox_ratio} -> {gearbox_ratio_updated}")
            df_calibration.loc[df_calibration['truck_id']==truck_id, 'gearbox_ratio'] = gearbox_ratio_updated
            
        if isinstance(motor_displacement_cm3_updated, Decimal) and (motor_displacement_cm3 != motor_displacement_cm3_updated):
            print(f"Truck: {registration}, motor displacement updated from {motor_displacement_cm3} -> {motor_displacement_cm3_updated}")
            df_calibration.loc[df_calibration['truck_id']==truck_id, 'motor_displacement_cm3'] = motor_displacement_cm3_updated
    
    
    # Save Cal to dynamo
    print('-------- Enable Writing after making sure Table is correct ----------')
    # if (len(df_calibration) == 105) and (df_calibration['truck_id'].unique().shape[0] == 105):
    #     wr.dynamodb.put_df(df_calibration, calibration_table_name)
    # else:
    #     print('Inconsistent data in dataframe')