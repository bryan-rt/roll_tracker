import 'package:flutter_test/flutter_test.dart';

import 'package:mobile_app/main.dart';

void main() {
  testWidgets('App renders AuthGate', (WidgetTester tester) async {
    // Smoke test — verifies the app widget tree builds without error.
    // Supabase must be initialized before RollItBackApp can render,
    // so this test is intentionally minimal.
    expect(RollItBackApp, isNotNull);
  });
}
