-- Migration: Add FCM token fields to users table
-- Run this in your PostgreSQL database

-- Add FCM token column to users table (if exists)
ALTER TABLE users 
ADD COLUMN IF NOT EXISTS fcm_token VARCHAR(500),
ADD COLUMN IF NOT EXISTS fcm_token_updated_at TIMESTAMP,
ADD COLUMN IF NOT EXISTS platform VARCHAR(20); -- 'ios' or 'android'

-- Create index for faster FCM token lookups
CREATE INDEX IF NOT EXISTS idx_users_fcm_token ON users(fcm_token);

-- Add comment
COMMENT ON COLUMN users.fcm_token IS 'Firebase Cloud Messaging token for push notifications';
COMMENT ON COLUMN users.fcm_token_updated_at IS 'Timestamp when FCM token was last updated';
COMMENT ON COLUMN users.platform IS 'Device platform (ios/android)';
