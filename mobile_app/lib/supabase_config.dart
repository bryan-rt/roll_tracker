import 'package:supabase_flutter/supabase_flutter.dart';

const String supabaseUrl = 'https://ugqfesjfyfacivpneazq.supabase.co';
const String supabaseKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVncWZlc2pmeWZhY2l2cG5lYXpxIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDgxMTA1OTUsImV4cCI6MjA2MzY4NjU5NX0.e8iBU_XDhJ0jOfUmHV64jMO-QcgV9e0QrcDxXwweqHo';

SupabaseClient get supabase => Supabase.instance.client;
