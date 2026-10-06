-- AlterTable
ALTER TABLE "ClassDefinition" ADD COLUMN "hasOcrStage" BOOLEAN NOT NULL DEFAULT false;

-- AlterTable
ALTER TABLE "Detection" ADD COLUMN "ocrLabel" TEXT,
ADD COLUMN "ocrText" TEXT,
ADD COLUMN "ocrConfidence" DOUBLE PRECISION,
ADD COLUMN "ocrManualCheck" BOOLEAN NOT NULL DEFAULT false,
ADD COLUMN "ocrModel" TEXT;
