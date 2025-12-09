-- =============================================
-- MES (Manufacturing Execution System) Database Schema
-- Focus: OEE (Overall Equipment Effectiveness) Tracking
-- Platform: Microsoft SQL Server
-- =============================================

USE [MES_OEE];
GO

-- =============================================
-- 1. CORE MANUFACTURING ENTITIES
-- =============================================

-- Manufacturing Sites/Plants
CREATE TABLE Sites (
    SiteID INT IDENTITY(1,1) PRIMARY KEY,
    SiteName NVARCHAR(100) NOT NULL,
    SiteCode NVARCHAR(10) NOT NULL UNIQUE,
    Location NVARCHAR(200),
    TimeZone NVARCHAR(50) DEFAULT 'UTC',
    IsActive BIT DEFAULT 1,
    CreatedDate DATETIME2 DEFAULT GETUTCDATE(),
    ModifiedDate DATETIME2 DEFAULT GETUTCDATE()
);

-- Production Areas/Departments
CREATE TABLE ProductionAreas (
    AreaID INT IDENTITY(1,1) PRIMARY KEY,
    SiteID INT NOT NULL FOREIGN KEY REFERENCES Sites(SiteID),
    AreaName NVARCHAR(100) NOT NULL,
    AreaCode NVARCHAR(10) NOT NULL,
    Description NVARCHAR(500),
    IsActive BIT DEFAULT 1,
    CreatedDate DATETIME2 DEFAULT GETUTCDATE(),
    CONSTRAINT UK_ProductionAreas_SiteArea UNIQUE (SiteID, AreaCode)
);

-- Production Lines
CREATE TABLE ProductionLines (
    LineID INT IDENTITY(1,1) PRIMARY KEY,
    AreaID INT NOT NULL FOREIGN KEY REFERENCES ProductionAreas(AreaID),
    LineName NVARCHAR(100) NOT NULL,
    LineCode NVARCHAR(20) NOT NULL,
    Description NVARCHAR(500),
    PlannedProductionTime INT NOT NULL DEFAULT 1440, -- Minutes per day
    IsActive BIT DEFAULT 1,
    CreatedDate DATETIME2 DEFAULT GETUTCDATE(),
    CONSTRAINT UK_ProductionLines_AreaLine UNIQUE (AreaID, LineCode)
);

-- Equipment/Machines
CREATE TABLE Equipment (
    EquipmentID INT IDENTITY(1,1) PRIMARY KEY,
    LineID INT NOT NULL FOREIGN KEY REFERENCES ProductionLines(LineID),
    EquipmentName NVARCHAR(100) NOT NULL,
    EquipmentCode NVARCHAR(20) NOT NULL,
    EquipmentType NVARCHAR(50) NOT NULL, -- 'Machine', 'Robot', 'Conveyor', etc.
    Manufacturer NVARCHAR(100),
    Model NVARCHAR(100),
    SerialNumber NVARCHAR(100),
    InstallDate DATE,
    MaxCapacityPerHour DECIMAL(10,2), -- Theoretical maximum output
    StandardCycleTime DECIMAL(10,3), -- Standard cycle time in minutes
    IsActive BIT DEFAULT 1,
    CreatedDate DATETIME2 DEFAULT GETUTCDATE(),
    ModifiedDate DATETIME2 DEFAULT GETUTCDATE(),
    CONSTRAINT UK_Equipment_LineCode UNIQUE (LineID, EquipmentCode)
);

-- Products
CREATE TABLE Products (
    ProductID INT IDENTITY(1,1) PRIMARY KEY,
    ProductCode NVARCHAR(50) NOT NULL UNIQUE,
    ProductName NVARCHAR(200) NOT NULL,
    ProductFamily NVARCHAR(100),
    Description NVARCHAR(500),
    StandardCycleTime DECIMAL(10,3), -- Minutes per unit
    TargetYield DECIMAL(5,4) DEFAULT 0.95, -- 95% default
    IsActive BIT DEFAULT 1,
    CreatedDate DATETIME2 DEFAULT GETUTCDATE()
);

-- Work Orders
CREATE TABLE WorkOrders (
    WorkOrderID INT IDENTITY(1,1) PRIMARY KEY,
    WorkOrderNumber NVARCHAR(50) NOT NULL UNIQUE,
    ProductID INT NOT NULL FOREIGN KEY REFERENCES Products(ProductID),
    LineID INT NOT NULL FOREIGN KEY REFERENCES ProductionLines(LineID),
    PlannedQuantity INT NOT NULL,
    PlannedStartTime DATETIME2 NOT NULL,
    PlannedEndTime DATETIME2 NOT NULL,
    ActualStartTime DATETIME2,
    ActualEndTime DATETIME2,
    Priority TINYINT DEFAULT 3, -- 1=High, 2=Medium, 3=Low
    Status NVARCHAR(20) DEFAULT 'Planned', -- Planned, Active, Completed, Cancelled
    CreatedDate DATETIME2 DEFAULT GETUTCDATE(),
    ModifiedDate DATETIME2 DEFAULT GETUTCDATE()
);

-- =============================================
-- 2. OEE CORE TRACKING TABLES
-- =============================================

-- Production Runs (detailed production tracking)
CREATE TABLE ProductionRuns (
    RunID INT IDENTITY(1,1) PRIMARY KEY,
    WorkOrderID INT NOT NULL FOREIGN KEY REFERENCES WorkOrders(WorkOrderID),
    EquipmentID INT NOT NULL FOREIGN KEY REFERENCES Equipment(EquipmentID),
    StartTime DATETIME2 NOT NULL,
    EndTime DATETIME2,
    PlannedQuantity INT NOT NULL,
    ActualQuantity INT DEFAULT 0,
    GoodQuantity INT DEFAULT 0,
    DefectQuantity INT DEFAULT 0,
    ScrapQuantity INT DEFAULT 0,
    ReworkQuantity INT DEFAULT 0,
    Status NVARCHAR(20) DEFAULT 'Active', -- Active, Completed, Aborted
    CreatedDate DATETIME2 DEFAULT GETUTCDATE(),
    ModifiedDate DATETIME2 DEFAULT GETUTCDATE()
);

-- Downtime Tracking (for Availability calculation)
CREATE TABLE DowntimeEvents (
    DowntimeID INT IDENTITY(1,1) PRIMARY KEY,
    EquipmentID INT NOT NULL FOREIGN KEY REFERENCES Equipment(EquipmentID),
    RunID INT FOREIGN KEY REFERENCES ProductionRuns(RunID),
    DowntimeReasonID INT NOT NULL, -- FK to DowntimeReasons
    StartTime DATETIME2 NOT NULL,
    EndTime DATETIME2,
    Duration AS DATEDIFF(MINUTE, StartTime, ISNULL(EndTime, GETUTCDATE())),
    PlannedDowntime BIT DEFAULT 0, -- TRUE for planned maintenance
    Description NVARCHAR(500),
    ReportedBy NVARCHAR(100),
    CreatedDate DATETIME2 DEFAULT GETUTCDATE(),
    ModifiedDate DATETIME2 DEFAULT GETUTCDATE()
);

-- Performance Events (speed losses, cycle time variations)
CREATE TABLE PerformanceEvents (
    PerformanceEventID INT IDENTITY(1,1) PRIMARY KEY,
    RunID INT NOT NULL FOREIGN KEY REFERENCES ProductionRuns(RunID),
    EquipmentID INT NOT NULL FOREIGN KEY REFERENCES Equipment(EquipmentID),
    EventTime DATETIME2 NOT NULL,
    ActualCycleTime DECIMAL(10,3), -- Minutes
    StandardCycleTime DECIMAL(10,3), -- Minutes
    QuantityProduced INT DEFAULT 1,
    PerformanceEfficiency AS (
        CASE
            WHEN ActualCycleTime > 0 AND StandardCycleTime > 0
            THEN (StandardCycleTime / ActualCycleTime) * 100
            ELSE NULL
        END
    ), -- Calculated performance %
    CreatedDate DATETIME2 DEFAULT GETUTCDATE()
);

-- Quality Events (defects, rework)
CREATE TABLE QualityEvents (
    QualityEventID INT IDENTITY(1,1) PRIMARY KEY,
    RunID INT NOT NULL FOREIGN KEY REFERENCES ProductionRuns(RunID),
    EquipmentID INT NOT NULL FOREIGN KEY REFERENCES Equipment(EquipmentID),
    DefectTypeID INT NOT NULL, -- FK to DefectTypes
    EventTime DATETIME2 NOT NULL,
    QuantityAffected INT NOT NULL DEFAULT 1,
    DefectSeverity NVARCHAR(20) DEFAULT 'Minor', -- Critical, Major, Minor
    ActionTaken NVARCHAR(20), -- Scrap, Rework, Accept
    RootCause NVARCHAR(500),
    DetectedBy NVARCHAR(100),
    CreatedDate DATETIME2 DEFAULT GETUTCDATE()
);

-- =============================================
-- 3. LOOKUP/REFERENCE TABLES
-- =============================================

-- Downtime Reason Categories
CREATE TABLE DowntimeCategories (
    CategoryID INT IDENTITY(1,1) PRIMARY KEY,
    CategoryName NVARCHAR(100) NOT NULL UNIQUE,
    CategoryCode NVARCHAR(10) NOT NULL UNIQUE,
    Description NVARCHAR(500),
    IsPlanned BIT DEFAULT 0, -- Planned vs Unplanned downtime
    CreatedDate DATETIME2 DEFAULT GETUTCDATE()
);

-- Specific Downtime Reasons
CREATE TABLE DowntimeReasons (
    ReasonID INT IDENTITY(1,1) PRIMARY KEY,
    CategoryID INT NOT NULL FOREIGN KEY REFERENCES DowntimeCategories(CategoryID),
    ReasonName NVARCHAR(100) NOT NULL,
    ReasonCode NVARCHAR(20) NOT NULL,
    Description NVARCHAR(500),
    IsActive BIT DEFAULT 1,
    CreatedDate DATETIME2 DEFAULT GETUTCDATE(),
    CONSTRAINT UK_DowntimeReasons_Code UNIQUE (ReasonCode)
);

-- Defect Types
CREATE TABLE DefectTypes (
    DefectTypeID INT IDENTITY(1,1) PRIMARY KEY,
    DefectName NVARCHAR(100) NOT NULL,
    DefectCode NVARCHAR(20) NOT NULL UNIQUE,
    Category NVARCHAR(50), -- Dimensional, Visual, Functional, etc.
    DefaultSeverity NVARCHAR(20) DEFAULT 'Minor',
    Description NVARCHAR(500),
    IsActive BIT DEFAULT 1,
    CreatedDate DATETIME2 DEFAULT GETUTCDATE()
);

-- Shifts
CREATE TABLE Shifts (
    ShiftID INT IDENTITY(1,1) PRIMARY KEY,
    SiteID INT NOT NULL FOREIGN KEY REFERENCES Sites(SiteID),
    ShiftName NVARCHAR(50) NOT NULL,
    StartTime TIME NOT NULL,
    EndTime TIME NOT NULL,
    IsActive BIT DEFAULT 1,
    CreatedDate DATETIME2 DEFAULT GETUTCDATE(),
    CONSTRAINT UK_Shifts_SiteName UNIQUE (SiteID, ShiftName)
);

-- =============================================
-- 4. OEE CALCULATION VIEWS
-- =============================================

-- Daily OEE Summary View
GO
CREATE VIEW vw_DailyOEE AS
WITH EquipmentDaily AS (
    SELECT
        e.EquipmentID,
        e.EquipmentName,
        e.EquipmentCode,
        pl.LineName,
        pa.AreaName,
        s.SiteName,
        CAST(pr.StartTime AS DATE) as ProductionDate,

        -- Availability Calculation
        SUM(DATEDIFF(MINUTE, pr.StartTime, ISNULL(pr.EndTime, GETUTCDATE()))) as PlannedProductionTime,
        ISNULL(SUM(de.Duration), 0) as UnplannedDowntime,
        (SUM(DATEDIFF(MINUTE, pr.StartTime, ISNULL(pr.EndTime, GETUTCDATE()))) - ISNULL(SUM(CASE WHEN de.PlannedDowntime = 0 THEN de.Duration ELSE 0 END), 0)) as OperatingTime,

        -- Performance Calculation
        SUM(pr.ActualQuantity) as ActualQuantity,
        SUM(pr.PlannedQuantity) as PlannedQuantity,
        AVG(pe.PerformanceEfficiency) as AvgPerformanceEfficiency,

        -- Quality Calculation
        SUM(pr.GoodQuantity) as GoodQuantity,
        SUM(pr.DefectQuantity + pr.ScrapQuantity) as DefectQuantity

    FROM Equipment e
    JOIN ProductionLines pl ON e.LineID = pl.LineID
    JOIN ProductionAreas pa ON pl.AreaID = pa.AreaID
    JOIN Sites s ON pa.SiteID = s.SiteID
    LEFT JOIN ProductionRuns pr ON e.EquipmentID = pr.EquipmentID
    LEFT JOIN DowntimeEvents de ON e.EquipmentID = de.EquipmentID
        AND CAST(de.StartTime AS DATE) = CAST(pr.StartTime AS DATE)
    LEFT JOIN PerformanceEvents pe ON pr.RunID = pe.RunID

    WHERE pr.StartTime >= DATEADD(DAY, -30, GETUTCDATE()) -- Last 30 days
    GROUP BY
        e.EquipmentID, e.EquipmentName, e.EquipmentCode,
        pl.LineName, pa.AreaName, s.SiteName,
        CAST(pr.StartTime AS DATE)
)
SELECT
    EquipmentID,
    EquipmentName,
    EquipmentCode,
    LineName,
    AreaName,
    SiteName,
    ProductionDate,

    -- Availability % = Operating Time / Planned Production Time
    CASE
        WHEN PlannedProductionTime > 0
        THEN ROUND((CAST(OperatingTime AS FLOAT) / PlannedProductionTime) * 100, 2)
        ELSE 0
    END as Availability,

    -- Performance % = (Actual Quantity / Planned Quantity) * 100
    CASE
        WHEN PlannedQuantity > 0
        THEN ROUND((CAST(ActualQuantity AS FLOAT) / PlannedQuantity) * 100, 2)
        ELSE ISNULL(AvgPerformanceEfficiency, 0)
    END as Performance,

    -- Quality % = Good Quantity / Total Quantity
    CASE
        WHEN (GoodQuantity + DefectQuantity) > 0
        THEN ROUND((CAST(GoodQuantity AS FLOAT) / (GoodQuantity + DefectQuantity)) * 100, 2)
        ELSE 100
    END as Quality,

    -- OEE % = Availability × Performance × Quality
    ROUND(
        (CASE WHEN PlannedProductionTime > 0 THEN (CAST(OperatingTime AS FLOAT) / PlannedProductionTime) ELSE 0 END) *
        (CASE WHEN PlannedQuantity > 0 THEN (CAST(ActualQuantity AS FLOAT) / PlannedQuantity) ELSE ISNULL(AvgPerformanceEfficiency, 0)/100 END) *
        (CASE WHEN (GoodQuantity + DefectQuantity) > 0 THEN (CAST(GoodQuantity AS FLOAT) / (GoodQuantity + DefectQuantity)) ELSE 1 END) * 100, 2
    ) as OEE,

    PlannedProductionTime,
    OperatingTime,
    UnplannedDowntime,
    ActualQuantity,
    PlannedQuantity,
    GoodQuantity,
    DefectQuantity

FROM EquipmentDaily
WHERE PlannedProductionTime > 0;

GO

-- =============================================
-- 5. INDEXES FOR PERFORMANCE
-- =============================================

-- Production Runs indexes
CREATE NONCLUSTERED INDEX IX_ProductionRuns_Equipment_Start ON ProductionRuns(EquipmentID, StartTime);
CREATE NONCLUSTERED INDEX IX_ProductionRuns_WorkOrder ON ProductionRuns(WorkOrderID);
CREATE NONCLUSTERED INDEX IX_ProductionRuns_Status_Start ON ProductionRuns(Status, StartTime);

-- Downtime Events indexes
CREATE NONCLUSTERED INDEX IX_DowntimeEvents_Equipment_Start ON DowntimeEvents(EquipmentID, StartTime);
CREATE NONCLUSTERED INDEX IX_DowntimeEvents_Reason_Start ON DowntimeEvents(DowntimeReasonID, StartTime);
CREATE NONCLUSTERED INDEX IX_DowntimeEvents_Planned_Start ON DowntimeEvents(PlannedDowntime, StartTime);

-- Performance Events indexes
CREATE NONCLUSTERED INDEX IX_PerformanceEvents_Run_Time ON PerformanceEvents(RunID, EventTime);
CREATE NONCLUSTERED INDEX IX_PerformanceEvents_Equipment_Time ON PerformanceEvents(EquipmentID, EventTime);

-- Quality Events indexes
CREATE NONCLUSTERED INDEX IX_QualityEvents_Run_Time ON QualityEvents(RunID, EventTime);
CREATE NONCLUSTERED INDEX IX_QualityEvents_DefectType_Time ON QualityEvents(DefectTypeID, EventTime);

-- Equipment lookup index
CREATE NONCLUSTERED INDEX IX_Equipment_Line_Active ON Equipment(LineID, IsActive);

-- Work Orders indexes
CREATE NONCLUSTERED INDEX IX_WorkOrders_Line_Status ON WorkOrders(LineID, Status);
CREATE NONCLUSTERED INDEX IX_WorkOrders_Product_Start ON WorkOrders(ProductID, PlannedStartTime);

-- =============================================
-- 6. SAMPLE DATA INSERTION
-- =============================================

-- Insert sample lookup data
INSERT INTO DowntimeCategories (CategoryName, CategoryCode, Description, IsPlanned) VALUES
('Equipment Failure', 'EF', 'Unplanned equipment breakdowns', 0),
('Setup/Changeover', 'SC', 'Product changeover and setup time', 1),
('Maintenance', 'PM', 'Planned preventive maintenance', 1),
('Material Shortage', 'MS', 'Lack of raw materials', 0),
('Quality Issues', 'QI', 'Quality-related stoppages', 0),
('Operator Break', 'OB', 'Scheduled operator breaks', 1);

INSERT INTO DowntimeReasons (CategoryID, ReasonName, ReasonCode, Description) VALUES
(1, 'Motor Failure', 'EF001', 'Electric motor malfunction'),
(1, 'Conveyor Jam', 'EF002', 'Product jam in conveyor system'),
(2, 'Tool Change', 'SC001', 'Cutting tool replacement'),
(2, 'Product Changeover', 'SC002', 'Change from one product to another'),
(3, 'Scheduled Maintenance', 'PM001', 'Regular preventive maintenance'),
(4, 'Raw Material Delay', 'MS001', 'Delayed delivery of raw materials'),
(5, 'Quality Check Fail', 'QI001', 'Failed quality inspection'),
(6, 'Lunch Break', 'OB001', 'Scheduled lunch break');

INSERT INTO DefectTypes (DefectName, DefectCode, Category, DefaultSeverity, Description) VALUES
('Dimensional Deviation', 'DIM001', 'Dimensional', 'Major', 'Part dimensions outside tolerance'),
('Surface Defect', 'VIS001', 'Visual', 'Minor', 'Cosmetic surface imperfection'),
('Assembly Error', 'ASM001', 'Functional', 'Critical', 'Incorrect assembly configuration'),
('Material Contamination', 'MAT001', 'Material', 'Major', 'Foreign material contamination');

-- Insert sample site and hierarchy
INSERT INTO Sites (SiteName, SiteCode, Location, TimeZone) VALUES
('Main Manufacturing Plant', 'MMP', 'Detroit, MI', 'America/Detroit');

INSERT INTO ProductionAreas (SiteID, AreaName, AreaCode, Description) VALUES
(1, 'Assembly Line 1', 'AL1', 'Primary assembly operations'),
(1, 'Quality Control', 'QC', 'Quality inspection and testing');

INSERT INTO ProductionLines (AreaID, LineName, LineCode, Description, PlannedProductionTime) VALUES
(1, 'Assembly Line A', 'ALA', 'Main product assembly line', 1200), -- 20 hours/day
(2, 'QC Station 1', 'QC1', 'Quality control station', 480); -- 8 hours/day

INSERT INTO Shifts (SiteID, ShiftName, StartTime, EndTime) VALUES
(1, 'Day Shift', '06:00:00', '14:00:00'),
(1, 'Evening Shift', '14:00:00', '22:00:00'),
(1, 'Night Shift', '22:00:00', '06:00:00');

PRINT 'MES OEE Database Schema created successfully!';
PRINT 'Key Features:';
PRINT '- Complete OEE tracking (Availability, Performance, Quality)';
PRINT '- Hierarchical equipment structure';
PRINT '- Detailed downtime and performance event tracking';
PRINT '- Quality control integration';
PRINT '- Performance-optimized indexes';
PRINT '- Real-time OEE calculation view';
PRINT '';
PRINT 'Next Steps:';
PRINT '1. Create specific equipment and product records';
PRINT '2. Set up data collection interfaces';
PRINT '3. Configure reporting dashboards';
PRINT '4. Implement alerting thresholds';