# MES OEE Database Implementation Guide

## Overview

This guide provides a comprehensive MSSQL database schema for a Manufacturing Execution System (MES) focused on Overall Equipment Effectiveness (OEE) tracking. The schema is designed for real-world manufacturing environments with scalability, performance, and data integrity in mind.

## Key Features

- ✅ **Complete OEE Calculation**: Tracks Availability, Performance, and Quality metrics
- ✅ **Hierarchical Equipment Structure**: Sites → Areas → Lines → Equipment
- ✅ **Real-time Data Collection**: Supports live production tracking
- ✅ **Comprehensive Downtime Tracking**: Planned vs unplanned downtime with categorization
- ✅ **Quality Management**: Defect tracking with root cause analysis
- ✅ **Performance Optimization**: Indexed for high-frequency data collection
- ✅ **Reporting Ready**: Pre-built views and procedures for dashboards

## OEE Calculation Formula

**OEE = Availability × Performance × Quality**

### Availability
```
Availability % = (Planned Production Time - Unplanned Downtime) / Planned Production Time × 100
```

### Performance
```
Performance % = (Actual Output / Planned Output) × 100
OR
Performance % = (Standard Cycle Time / Actual Cycle Time) × 100
```

### Quality
```
Quality % = Good Units / Total Units × 100
```

## Database Schema Structure

### Core Entities Hierarchy

```
Sites (Manufacturing Plants)
├── ProductionAreas (Departments/Areas)
│   ├── ProductionLines (Assembly Lines)
│   │   ├── Equipment (Machines/Stations)
│   │   │   ├── ProductionRuns (Production Events)
│   │   │   ├── DowntimeEvents (Downtime Tracking)
│   │   │   ├── PerformanceEvents (Speed/Efficiency)
│   │   │   └── QualityEvents (Defects/Issues)
```

### Key Tables

| Table | Purpose | Key Fields |
|-------|---------|------------|
| **Sites** | Manufacturing locations | SiteID, SiteName, Location |
| **Equipment** | Machines and stations | EquipmentID, LineID, MaxCapacity |
| **ProductionRuns** | Production tracking | RunID, StartTime, ActualQuantity, GoodQuantity |
| **DowntimeEvents** | Downtime logging | DowntimeID, StartTime, Duration, ReasonID |
| **QualityEvents** | Defect tracking | QualityEventID, DefectTypeID, QuantityAffected |
| **WorkOrders** | Production orders | WorkOrderID, ProductID, PlannedQuantity |

## Implementation Steps

### Phase 1: Database Setup

1. **Create Database**
```sql
CREATE DATABASE [MES_OEE];
GO
USE [MES_OEE];
GO
```

2. **Run Schema Scripts**
```sql
-- Execute MES_OEE_Schema.sql
-- Execute MES_OEE_StoredProcedures.sql
```

3. **Set up Initial Data**
```sql
-- Sites, Areas, Lines
-- Equipment configuration
-- Products and standard cycle times
-- Downtime reasons and categories
-- Defect types
```

### Phase 2: Data Collection Integration

#### Real-time Production Data
```sql
-- Record production every cycle/batch
EXEC sp_RecordProduction
    @WorkOrderID = 1001,
    @EquipmentID = 5,
    @QuantityProduced = 10,
    @GoodQuantity = 9,
    @DefectQuantity = 1,
    @CycleTime = 2.5;
```

#### Downtime Tracking
```sql
-- Start downtime
EXEC sp_RecordDowntime
    @EquipmentID = 5,
    @DowntimeReasonID = 1, -- Equipment Failure
    @StartTime = GETUTCDATE(),
    @Description = 'Motor overheating',
    @ReportedBy = 'Operator123';

-- End downtime
EXEC sp_CloseDowntime
    @DowntimeID = 1,
    @EndTime = GETUTCDATE(),
    @AdditionalNotes = 'Motor replaced';
```

### Phase 3: Reporting and Analytics

#### Real-time OEE Dashboard
```sql
-- Get current line performance
EXEC sp_GetLineDashboard @LineID = 1, @Date = '2024-01-15';

-- Get equipment OEE for specific period
EXEC sp_GetEquipmentOEE
    @EquipmentID = 5,
    @StartDate = '2024-01-15 00:00:00',
    @EndDate = '2024-01-15 23:59:59';
```

#### Loss Analysis
```sql
-- Identify top loss sources
EXEC sp_GetTopLosses
    @LineID = 1,
    @StartDate = '2024-01-01',
    @EndDate = '2024-01-31',
    @TopN = 10;
```

## Performance Considerations

### Indexing Strategy
The schema includes optimized indexes for:
- Time-based queries (most common in manufacturing)
- Equipment-based filtering
- Status-based filtering
- Foreign key relationships

### Data Retention
Consider implementing data archiving for:
- Production events older than 2 years
- Closed downtime events older than 1 year
- Performance events (high volume) older than 6 months

### Partitioning (For High Volume)
For sites with high data volume, consider table partitioning:
```sql
-- Partition ProductionRuns by month
CREATE PARTITION FUNCTION pf_ProductionMonth (DATETIME2)
AS RANGE RIGHT FOR VALUES
('2024-01-01', '2024-02-01', '2024-03-01', ...);
```

## Integration Points

### Data Collection Sources
- **PLC/SCADA Systems**: Equipment status, cycle times, counts
- **Operator Terminals**: Downtime reasons, quality issues
- **Vision Systems**: Quality inspection results
- **ERP Systems**: Work orders, planned quantities
- **CMMS Systems**: Maintenance schedules

### Common Integration Patterns

#### REST API Example
```csharp
// C# example for recording production
public async Task<bool> RecordProduction(ProductionEvent evt)
{
    using var connection = new SqlConnection(connectionString);
    var result = await connection.QueryAsync(
        "sp_RecordProduction",
        new {
            WorkOrderID = evt.WorkOrderId,
            EquipmentID = evt.EquipmentId,
            QuantityProduced = evt.Quantity,
            GoodQuantity = evt.GoodQuantity,
            CycleTime = evt.CycleTime
        },
        commandType: CommandType.StoredProcedure);

    return result.Any();
}
```

#### Message Queue Integration
```python
# Python example for async data processing
import pyodbc
import json
from azure.servicebus import ServiceBusClient

def process_production_message(message_body):
    data = json.loads(message_body)

    cursor.execute("""
        EXEC sp_RecordProduction
        @WorkOrderID=?, @EquipmentID=?, @QuantityProduced=?,
        @GoodQuantity=?, @CycleTime=?
    """, data['work_order'], data['equipment'],
         data['quantity'], data['good_qty'], data['cycle_time'])
```

## Monitoring and Alerting

### Key Metrics to Monitor
- **OEE Thresholds**: Alert when OEE drops below 60%
- **Downtime Duration**: Alert on downtimes > 30 minutes
- **Quality Issues**: Alert on defect rates > 5%
- **Performance Drops**: Alert when efficiency < 80%

### Sample Alert Queries
```sql
-- Equipment with low OEE (last 24 hours)
SELECT * FROM vw_DailyOEE
WHERE ProductionDate >= DATEADD(DAY, -1, GETDATE())
  AND OEE < 60
  ORDER BY OEE ASC;

-- Active downtimes over 30 minutes
SELECT
    e.EquipmentName,
    dr.ReasonName,
    de.StartTime,
    DATEDIFF(MINUTE, de.StartTime, GETUTCDATE()) as MinutesDown
FROM DowntimeEvents de
JOIN Equipment e ON de.EquipmentID = e.EquipmentID
JOIN DowntimeReasons dr ON de.DowntimeReasonID = dr.ReasonID
WHERE de.EndTime IS NULL
  AND DATEDIFF(MINUTE, de.StartTime, GETUTCDATE()) > 30;
```

## Best Practices

### Data Quality
1. **Validation Rules**: Implement constraints to prevent invalid data
2. **Audit Trail**: Track who enters data and when
3. **Data Cleansing**: Regular cleanup of incomplete records

### Security
1. **Role-Based Access**: Separate read/write permissions
2. **Sensitive Data**: Encrypt any employee/customer data
3. **Audit Logging**: Track all data modifications

### Maintenance
1. **Index Maintenance**: Regular index rebuilds/reorganization
2. **Statistics Updates**: Keep query optimizer statistics current
3. **Backup Strategy**: Regular backups with point-in-time recovery

## Typical OEE Benchmarks

| Industry | World Class OEE | Good OEE | Typical OEE |
|----------|-----------------|----------|-------------|
| Automotive | 85%+ | 70-84% | 50-69% |
| Electronics | 85%+ | 65-84% | 45-64% |
| Food & Beverage | 85%+ | 65-84% | 40-64% |
| Pharmaceutical | 85%+ | 70-84% | 55-69% |
| Chemical | 85%+ | 60-84% | 40-59% |

## Troubleshooting Common Issues

### Performance Problems
- **Symptom**: Slow OEE calculations
- **Solution**: Check indexes, consider pre-calculated summaries

### Data Integrity Issues
- **Symptom**: Missing production data
- **Solution**: Implement data validation, audit incomplete records

### Integration Failures
- **Symptom**: Data not updating from shop floor
- **Solution**: Check service connections, implement retry logic

## Next Steps

1. **Pilot Implementation**: Start with one production line
2. **Data Validation**: Compare calculated OEE with manual calculations
3. **User Training**: Train operators on data entry procedures
4. **Reporting Setup**: Create dashboards and regular reports
5. **Continuous Improvement**: Use data to identify improvement opportunities

## Support and Extensions

### Common Extensions
- **Predictive Maintenance**: Add equipment health monitoring
- **Energy Monitoring**: Track power consumption
- **Labor Tracking**: Connect with workforce management
- **Supply Chain**: Integration with inventory systems

### Advanced Analytics
- **Machine Learning**: Predictive failure detection
- **Statistical Process Control**: Quality trend analysis
- **Optimization**: Production scheduling optimization

---

This schema provides a solid foundation for manufacturing OEE tracking and can be extended based on specific business requirements.