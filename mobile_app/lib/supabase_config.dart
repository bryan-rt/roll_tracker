import 'package:supabase_flutter/supabase_flutter.dart';

const String supabaseUrl = 'http://192.168.0.66:54321';
const String supabaseKey = 'sb_publishable_ACJWlzQHlZjBrEguHvfOxg_3BJgxAaH';

SupabaseClient get supabase => Supabase.instance.client;
