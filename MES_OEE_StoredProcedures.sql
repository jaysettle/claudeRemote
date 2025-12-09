-- =============================================
-- MES OEE System - Stored Procedures
-- =============================================

USE [MES_OEE];
GO

-- =============================================
-- Procedure: Calculate Real-time OEE for Equipment
-- =============================================
CREATE OR ALTER PROCEDURE sp_GetEquipmentOEE
    @EquipmentID INT,
    @StartDate DATETIME2,
    @EndDate DATETIME2
AS
BEGIN
    SET NOCOUNT ON;

    DECLARE @PlannedTime INT = 0;
    DECLARE @DownTime INT = 0;
    DECLARE @OperatingTime INT = 0;
    DECLARE @ActualQuantity INT = 0;
    DECLARE @PlannedQuantity INT = 0;
    DECLARE @GoodQuantity INT = 0;
    DECLARE @TotalQuantity INT = 0;

    -- Get planned production time in minutes
    SELECT @PlannedTime = DATEDIFF(MINUTE, @StartDate, @EndDate);

    -- Get total downtime (unplanned only)
    SELECT @DownTime = ISNULL(SUM(DATEDIFF(MINUTE, StartTime, ISNULL(EndTime, GETUTCDATE()))), 0)
    FROM DowntimeEvents de
    JOIN DowntimeReasons dr ON de.DowntimeReasonID = dr.ReasonID
    JOIN DowntimeCategories dc ON dr.CategoryID = dc.CategoryID
    WHERE de.EquipmentID = @EquipmentID
      AND de.StartTime >= @StartDate
      AND de.StartTime < @EndDate
      AND dc.IsPlanned = 0; -- Only unplanned downtime affects availability

    -- Calculate operating time
    SET @OperatingTime = @PlannedTime - @DownTime;

    -- Get production quantities
    SELECT
        @ActualQuantity = ISNULL(SUM(ActualQuantity), 0),
        @PlannedQuantity = ISNULL(SUM(PlannedQuantity), 0),
        @GoodQuantity = ISNULL(SUM(GoodQuantity), 0),
        @TotalQuantity = ISNULL(SUM(ActualQuantity), 0)
    FROM ProductionRuns
    WHERE EquipmentID = @EquipmentID
      AND StartTime >= @StartDate
      AND StartTime < @EndDate;

    -- Calculate OEE components
    DECLARE @Availability DECIMAL(5,2) = CASE WHEN @PlannedTime > 0 THEN ROUND((@OperatingTime * 100.0) / @PlannedTime, 2) ELSE 0 END;
    DECLARE @Performance DECIMAL(5,2) = CASE WHEN @PlannedQuantity > 0 THEN ROUND((@ActualQuantity * 100.0) / @PlannedQuantity, 2) ELSE 0 END;
    DECLARE @Quality DECIMAL(5,2) = CASE WHEN @TotalQuantity > 0 THEN ROUND((@GoodQuantity * 100.0) / @TotalQuantity, 2) ELSE 100 END;
    DECLARE @OEE DECIMAL(5,2) = ROUND((@Availability * @Performance * @Quality) / 10000, 2);

    -- Return results
    SELECT
        @EquipmentID as EquipmentID,
        e.EquipmentName,
        e.EquipmentCode,
        @StartDate as PeriodStart,
        @EndDate as PeriodEnd,
        @PlannedTime as PlannedTimeMinutes,
        @DownTime as DowntimeMinutes,
        @OperatingTime as OperatingTimeMinutes,
        @ActualQuantity as ActualQuantity,
        @PlannedQuantity as PlannedQuantity,
        @GoodQuantity as GoodQuantity,
        @TotalQuantity as TotalQuantity,
        @Availability as Availability,
        @Performance as Performance,
        @Quality as Quality,
        @OEE as OEE,
        CASE
            WHEN @OEE >= 85 THEN 'World Class'
            WHEN @OEE >= 60 THEN 'Good'
            WHEN @OEE >= 40 THEN 'Fair'
            ELSE 'Poor'
        END as OEERating
    FROM Equipment e
    WHERE e.EquipmentID = @EquipmentID;
END;
GO

-- =============================================
-- Procedure: Get Top Loss Categories
-- =============================================
CREATE OR ALTER PROCEDURE sp_GetTopLosses
    @EquipmentID INT = NULL,
    @LineID INT = NULL,
    @StartDate DATETIME2,
    @EndDate DATETIME2,
    @TopN INT = 10
AS
BEGIN
    SET NOCOUNT ON;

    -- Downtime Losses
    SELECT TOP (@TopN)
        'Downtime' as LossType,
        dc.CategoryName as LossCategory,
        dr.ReasonName as LossReason,
        COUNT(*) as EventCount,
        SUM(DATEDIFF(MINUTE, de.StartTime, ISNULL(de.EndTime, GETUTCDATE()))) as TotalMinutes,
        ROUND(SUM(DATEDIFF(MINUTE, de.StartTime, ISNULL(de.EndTime, GETUTCDATE()))) / 60.0, 2) as TotalHours,
        ROUND(AVG(CAST(DATEDIFF(MINUTE, de.StartTime, ISNULL(de.EndTime, GETUTCDATE())) AS FLOAT)), 2) as AvgMinutesPerEvent
    FROM DowntimeEvents de
    JOIN DowntimeReasons dr ON de.DowntimeReasonID = dr.ReasonID
    JOIN DowntimeCategories dc ON dr.CategoryID = dc.CategoryID
    JOIN Equipment e ON de.EquipmentID = e.EquipmentID
    WHERE de.StartTime >= @StartDate
      AND de.StartTime < @EndDate
      AND (@EquipmentID IS NULL OR de.EquipmentID = @EquipmentID)
      AND (@LineID IS NULL OR e.LineID = @LineID)
      AND dc.IsPlanned = 0 -- Focus on unplanned losses
    GROUP BY dc.CategoryName, dr.ReasonName
    ORDER BY TotalMinutes DESC;

    -- Quality Losses
    SELECT TOP (@TopN)
        'Quality' as LossType,
        dt.Category as LossCategory,
        dt.DefectName as LossReason,
        COUNT(*) as EventCount,
        SUM(qe.QuantityAffected) as TotalQuantityAffected,
        NULL as TotalMinutes,
        NULL as TotalHours,
        ROUND(AVG(CAST(qe.QuantityAffected AS FLOAT)), 2) as AvgQuantityPerEvent
    FROM QualityEvents qe
    JOIN DefectTypes dt ON qe.DefectTypeID = dt.DefectTypeID
    JOIN ProductionRuns pr ON qe.RunID = pr.RunID
    JOIN Equipment e ON pr.EquipmentID = e.EquipmentID
    WHERE qe.EventTime >= @StartDate
      AND qe.EventTime < @EndDate
      AND (@EquipmentID IS NULL OR e.EquipmentID = @EquipmentID)
      AND (@LineID IS NULL OR e.LineID = @LineID)
    GROUP BY dt.Category, dt.DefectName
    ORDER BY TotalQuantityAffected DESC;
END;
GO

-- =============================================
-- Procedure: Record Production Event
-- =============================================
CREATE OR ALTER PROCEDURE sp_RecordProduction
    @WorkOrderID INT,
    @EquipmentID INT,
    @QuantityProduced INT,
    @GoodQuantity INT,
    @DefectQuantity INT = 0,
    @ScrapQuantity INT = 0,
    @ReworkQuantity INT = 0,
    @CycleTime DECIMAL(10,3) = NULL,
    @EventTime DATETIME2 = NULL
AS
BEGIN
    SET NOCOUNT ON;
    SET @EventTime = ISNULL(@EventTime, GETUTCDATE());

    DECLARE @RunID INT;
    DECLARE @StandardCycleTime DECIMAL(10,3);

    -- Get or create active production run
    SELECT @RunID = RunID
    FROM ProductionRuns
    WHERE WorkOrderID = @WorkOrderID
      AND EquipmentID = @EquipmentID
      AND Status = 'Active'
      AND EndTime IS NULL;

    -- If no active run, create one
    IF @RunID IS NULL
    BEGIN
        INSERT INTO ProductionRuns (WorkOrderID, EquipmentID, StartTime, PlannedQuantity, Status)
        SELECT @WorkOrderID, @EquipmentID, @EventTime, wo.PlannedQuantity, 'Active'
        FROM WorkOrders wo
        WHERE wo.WorkOrderID = @WorkOrderID;

        SET @RunID = SCOPE_IDENTITY();
    END

    -- Update production run totals
    UPDATE ProductionRuns
    SET ActualQuantity = ActualQuantity + @QuantityProduced,
        GoodQuantity = GoodQuantity + @GoodQuantity,
        DefectQuantity = DefectQuantity + @DefectQuantity,
        ScrapQuantity = ScrapQuantity + @ScrapQuantity,
        ReworkQuantity = ReworkQuantity + @ReworkQuantity,
        ModifiedDate = @EventTime
    WHERE RunID = @RunID;

    -- Record performance event if cycle time provided
    IF @CycleTime IS NOT NULL
    BEGIN
        -- Get standard cycle time
        SELECT @StandardCycleTime = p.StandardCycleTime
        FROM WorkOrders wo
        JOIN Products p ON wo.ProductID = p.ProductID
        WHERE wo.WorkOrderID = @WorkOrderID;

        INSERT INTO PerformanceEvents (RunID, EquipmentID, EventTime, ActualCycleTime, StandardCycleTime, QuantityProduced)
        VALUES (@RunID, @EquipmentID, @EventTime, @CycleTime, @StandardCycleTime, @QuantityProduced);
    END

    SELECT @RunID as RunID, 'Production recorded successfully' as Message;
END;
GO

-- =============================================
-- Procedure: Record Downtime Event
-- =============================================
CREATE OR ALTER PROCEDURE sp_RecordDowntime
    @EquipmentID INT,
    @DowntimeReasonID INT,
    @StartTime DATETIME2,
    @EndTime DATETIME2 = NULL,
    @Description NVARCHAR(500) = NULL,
    @ReportedBy NVARCHAR(100) = NULL
AS
BEGIN
    SET NOCOUNT ON;

    DECLARE @DowntimeID INT;

    INSERT INTO DowntimeEvents (EquipmentID, DowntimeReasonID, StartTime, EndTime, Description, ReportedBy)
    VALUES (@EquipmentID, @DowntimeReasonID, @StartTime, @EndTime, @Description, @ReportedBy);

    SET @DowntimeID = SCOPE_IDENTITY();

    SELECT
        @DowntimeID as DowntimeID,
        CASE WHEN @EndTime IS NULL THEN 'Downtime started' ELSE 'Downtime recorded' END as Message,
        CASE WHEN @EndTime IS NULL THEN NULL ELSE DATEDIFF(MINUTE, @StartTime, @EndTime) END as DurationMinutes;
END;
GO

-- =============================================
-- Procedure: Close Downtime Event
-- =============================================
CREATE OR ALTER PROCEDURE sp_CloseDowntime
    @DowntimeID INT,
    @EndTime DATETIME2 = NULL,
    @AdditionalNotes NVARCHAR(500) = NULL
AS
BEGIN
    SET NOCOUNT ON;
    SET @EndTime = ISNULL(@EndTime, GETUTCDATE());

    UPDATE DowntimeEvents
    SET EndTime = @EndTime,
        Description = CASE
            WHEN @AdditionalNotes IS NOT NULL
            THEN ISNULL(Description, '') + CHAR(13) + CHAR(10) + @AdditionalNotes
            ELSE Description
        END,
        ModifiedDate = GETUTCDATE()
    WHERE DowntimeID = @DowntimeID
      AND EndTime IS NULL;

    SELECT
        @DowntimeID as DowntimeID,
        'Downtime closed' as Message,
        DATEDIFF(MINUTE, StartTime, @EndTime) as TotalMinutes
    FROM DowntimeEvents
    WHERE DowntimeID = @DowntimeID;
END;
GO

-- =============================================
-- Procedure: Get Line Performance Dashboard
-- =============================================
CREATE OR ALTER PROCEDURE sp_GetLineDashboard
    @LineID INT,
    @Date DATE = NULL
AS
BEGIN
    SET NOCOUNT ON;
    SET @Date = ISNULL(@Date, CAST(GETUTCDATE() AS DATE));

    DECLARE @StartDate DATETIME2 = @Date;
    DECLARE @EndDate DATETIME2 = DATEADD(DAY, 1, @Date);

    -- Line Summary
    SELECT
        pl.LineID,
        pl.LineName,
        pl.LineCode,
        COUNT(DISTINCT e.EquipmentID) as TotalEquipment,
        COUNT(DISTINCT CASE WHEN pr.RunID IS NOT NULL THEN e.EquipmentID END) as ActiveEquipment,
        ROUND(AVG(oee.OEE), 2) as AvgOEE,
        ROUND(AVG(oee.Availability), 2) as AvgAvailability,
        ROUND(AVG(oee.Performance), 2) as AvgPerformance,
        ROUND(AVG(oee.Quality), 2) as AvgQuality,
        SUM(oee.ActualQuantity) as TotalProduction,
        SUM(oee.GoodQuantity) as TotalGoodQuantity,
        SUM(oee.DefectQuantity) as TotalDefects
    FROM ProductionLines pl
    LEFT JOIN Equipment e ON pl.LineID = e.LineID AND e.IsActive = 1
    LEFT JOIN vw_DailyOEE oee ON e.EquipmentID = oee.EquipmentID AND oee.ProductionDate = @Date
    LEFT JOIN ProductionRuns pr ON e.EquipmentID = pr.EquipmentID
        AND pr.StartTime >= @StartDate AND pr.StartTime < @EndDate AND pr.Status = 'Active'
    WHERE pl.LineID = @LineID
    GROUP BY pl.LineID, pl.LineName, pl.LineCode;

    -- Equipment Details
    SELECT
        e.EquipmentID,
        e.EquipmentName,
        e.EquipmentCode,
        CASE
            WHEN pr.RunID IS NOT NULL THEN 'Running'
            WHEN de.DowntimeID IS NOT NULL THEN 'Down'
            ELSE 'Idle'
        END as Status,
        ISNULL(oee.OEE, 0) as OEE,
        ISNULL(oee.Availability, 0) as Availability,
        ISNULL(oee.Performance, 0) as Performance,
        ISNULL(oee.Quality, 0) as Quality,
        ISNULL(oee.ActualQuantity, 0) as DailyProduction,
        wo.WorkOrderNumber as CurrentWorkOrder,
        p.ProductName as CurrentProduct
    FROM Equipment e
    LEFT JOIN vw_DailyOEE oee ON e.EquipmentID = oee.EquipmentID AND oee.ProductionDate = @Date
    LEFT JOIN ProductionRuns pr ON e.EquipmentID = pr.EquipmentID
        AND pr.StartTime >= @StartDate AND pr.StartTime < @EndDate AND pr.Status = 'Active'
    LEFT JOIN WorkOrders wo ON pr.WorkOrderID = wo.WorkOrderID
    LEFT JOIN Products p ON wo.ProductID = p.ProductID
    LEFT JOIN DowntimeEvents de ON e.EquipmentID = de.EquipmentID
        AND de.StartTime >= @StartDate AND de.StartTime < @EndDate AND de.EndTime IS NULL
    WHERE e.LineID = @LineID AND e.IsActive = 1
    ORDER BY e.EquipmentName;
END;
GO

-- =============================================
-- Example Usage Script
-- =============================================

PRINT 'MES OEE Stored Procedures created successfully!';
PRINT '';
PRINT 'Available Procedures:';
PRINT '- sp_GetEquipmentOEE: Calculate OEE for specific equipment and time period';
PRINT '- sp_GetTopLosses: Identify biggest loss sources (downtime and quality)';
PRINT '- sp_RecordProduction: Record production events with quantities';
PRINT '- sp_RecordDowntime: Start downtime tracking';
PRINT '- sp_CloseDowntime: End downtime events';
PRINT '- sp_GetLineDashboard: Get real-time line performance dashboard';
PRINT '';
PRINT 'Example Usage:';
PRINT 'EXEC sp_GetEquipmentOEE @EquipmentID=1, @StartDate=''2024-01-01'', @EndDate=''2024-01-02'';';
PRINT 'EXEC sp_GetLineDashboard @LineID=1, @Date=''2024-01-01'';';