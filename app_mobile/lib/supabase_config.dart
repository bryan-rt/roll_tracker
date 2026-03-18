import 'package:supabase_flutter/supabase_flutter.dart';

// Remote Supabase (production)
const String supabaseUrl = 'https://zwwdduccwrkmkvawwjpc.supabase.co';
const String supabaseKey = 'sb_publishable_-RDxC1o-1dIa8WSaNCjTVQ_nM1GPdnI';

// Local Supabase (dev — uncomment to switch)
// const String supabaseUrl = 'http://192.168.0.66:54321';
// const String supabaseKey = 'sb_publishable_ACJWlzQHlZjBrEguHvfOxg_3BJgxAaH';

SupabaseClient get supabase => Supabase.instance.client;
